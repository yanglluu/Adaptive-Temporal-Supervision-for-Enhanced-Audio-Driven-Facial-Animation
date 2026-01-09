# losses/stft_stable.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------
# helpers
# -------------------------

def _hann(n, device):
    """Create a Hann window on the given device."""
    return torch.hann_window(n, periodic=True, device=device)

def _mag(spec, eps=1e-6):
    """Safe magnitude with clamp to avoid log(0) and division by zero."""
    m = spec.abs()
    m = torch.clamp(m, min=eps)
    # 再做一次兜底，清理潜在 NaN/Inf
    m = torch.nan_to_num(m, nan=eps, posinf=1e6, neginf=eps)
    return m

@torch.no_grad()
def _safe_weight_ema(new_w,
                     old_w=None,              # type: Optional[torch.Tensor]
                     ema=0.8,
                     wmin=0.5,
                     wmax=1.2):
    """
    Clamp -> EMA -> return (detach handled by caller if needed).
    new_w, old_w: [B*, nb]
    """
    w = torch.clamp(new_w, wmin, wmax)
    if old_w is None:
        return w
    return ema * old_w + (1.0 - ema) * w


def _adapt_input_to_bt(x):
    """
    Accept [B, T] or [B, T, C]; return as [B*C, T].
    Last dim must be time after reshape.
    """
    if x.dim() == 2:
        # [B, T]
        return x
    if x.dim() == 3:
        # [B, T, C] -> [B*C, T]
        B, T, C = x.shape
        return x.transpose(1, 2).reshape(-1, T).contiguous()
    raise ValueError("Expect x dim=2/3, got {}".format(x.dim()))


def _choose_stft_params(T, n_fft_cfg, hop_cfg, win_cfg):
    """
    根据时间长度 T 自适应 n_fft/hop/win，并决定是否使用 center=True。
    目标：确保 center=True 时 pad (= n_fft//2) < T；否则改 center=False。
    Returns:
        (n_fft_eff, hop_eff, win_eff, use_center)  # type: Tuple[int,int,int,bool]
    """
    # 极短序列直接早退由上层处理
    n_fft_eff = min(n_fft_cfg, max(32, 2 * ((T // 2) - 1)))  # 保证 n_fft_eff//2 < T
    # hop 保守在 [n_fft/4, n_fft/2] 内
    hop_eff   = max(8, min(hop_cfg, n_fft_eff // 2))
    hop_eff   = max(hop_eff, n_fft_eff // 4)
    win_eff   = min(win_cfg, n_fft_eff)

    use_center = True
    if (n_fft_eff // 2) >= T:
        use_center = False

    return n_fft_eff, hop_eff, win_eff, use_center


# -------------------------
# Ultra-short STFT (AU45)
# -------------------------

class UltraShortWindowSTFTLossStable(nn.Module):
    """
    超短窗 STFT（默认 n_fft=32, hop=8, win=32），用于 AU45/眼周等极短时结构。
    特性：
      - 强制 float32 计算（禁用 AMP）
      - clamp + log1p 压动态范围
      - 支持 [B, T] / [B, T, C] 输入（内部统一成 [B*C, T]）
    """
    def __init__(self,
                 n_fft=32,
                 hop_length=8,
                 win_length=32,
                 weight=0.0):
        super(UltraShortWindowSTFTLossStable, self).__init__()
        self.n_fft = n_fft
        self.hop   = hop_length
        self.win   = win_length
        self.weight = weight

    def forward(self, x_pred, x_gt):
        if self.weight <= 0:
            return torch.zeros([], device=x_pred.device)

        # 统一为 [B*C, T]
        x_pred = _adapt_input_to_bt(x_pred)
        x_gt   = _adapt_input_to_bt(x_gt)

        # 序列长度
        T = x_pred.shape[-1]
        if T < 16:
            # 极短序列不计算 STFT，返回 0
            return torch.zeros([], device=x_pred.device)

        with torch.cuda.amp.autocast(enabled=False):
            dev = x_pred.device
            n_fft_eff, hop_eff, win_eff, use_center = _choose_stft_params(
                T, self.n_fft, self.hop, self.win
            )
            win = _hann(win_eff, dev)

            # 安全输入
            x_pred = torch.nan_to_num(x_pred.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            x_gt   = torch.nan_to_num(x_gt.float(),   nan=0.0, posinf=1e4, neginf=-1e4)

            spec_p = torch.stft(x_pred, n_fft=n_fft_eff, hop_length=hop_eff,
                                win_length=win_eff, window=win, center=use_center,
                                return_complex=True)
            spec_g = torch.stft(x_gt,   n_fft=n_fft_eff, hop_length=hop_eff,
                                win_length=win_eff, window=win, center=use_center,
                                return_complex=True)

            mag_p = _mag(spec_p)  # [B*, F, T']
            mag_g = _mag(spec_g)

            # 使用 log1p 压动态范围，更稳
            loss = F.l1_loss(torch.log1p(mag_p), torch.log1p(mag_g))

        return self.weight * loss


# -------------------------
# Banded STFT (3 bands)
# -------------------------

class BandedSTFTLossStable(nn.Module):
    """
    分带 STFT（默认三段：低/中/高），带自适应高频加权、EMA 护栏、warmup。
    - 输入支持 [B, T] 或 [B, T, C]，内部统一为 [B*, T]。
    - 动态 n_fft/hop/win，确保 pad < T；必要时自动 center=False。
    - 幅度谱做 clamp + nan_to_num，避免 NaN/Inf。
    - band 权重做 EMA（wmin/wmax 护栏）与逐样本归一（+1e-6）。
    """
    def __init__(self,
                 n_fft=128,
                 hop_length=32,
                 win_length=96,
                 bands=((0, 64), (64, 96), (96, 128)),
                 band_weights=(0.05, 0.05, 0.10),
                 adaptive=True,
                 ema=0.8,
                 wmin=0.5,
                 wmax=1.2,
                 weight=0.18,
                 warmup_steps=1000):
        super(BandedSTFTLossStable, self).__init__()
        self.n_fft  = n_fft
        self.hop    = hop_length
        self.win    = win_length
        self.bands  = bands
        self.register_buffer("base_w", torch.tensor(band_weights, dtype=torch.float32))
        self.adaptive = adaptive
        self.ema      = ema
        self.wmin     = wmin
        self.wmax     = wmax
        self.weight   = weight
        self.warmup_steps = warmup_steps

        # 运行态缓存
        self.register_buffer("_ewma_w", None)                       # EMA 权重缓存 [B*, nb]
        self.register_buffer("_steps",  torch.tensor(0, dtype=torch.long))

    def _band_energy(self, mag):
        """
        mag: [B*, F, T'] (magnitude)
        return: [B*, nb]  每个 band 的能量（log1p 在 forward 里加）
        """
        energies = []
        for (lo, hi) in self.bands:
            energies.append(mag[:, lo:hi].mean(dim=(1, 2)))
        return torch.stack(energies, dim=-1)  # [B*, nb]

    def forward(self, x_pred, x_gt):
        if self.weight <= 0:
            return torch.zeros([], device=x_pred.device)

        # 统一为 [B*, T]
        x_pred = _adapt_input_to_bt(x_pred)
        x_gt   = _adapt_input_to_bt(x_gt)

        # 长度太短直接早退
        T = x_pred.shape[-1]
        if T < 32:
            return torch.zeros([], device=x_pred.device)

        with torch.cuda.amp.autocast(enabled=False):
            dev = x_pred.device

            # 自适应 n_fft/hop/win + center
            n_fft_eff, hop_eff, win_eff, use_center = _choose_stft_params(
                T, self.n_fft, self.hop, self.win
            )
            win = _hann(win_eff, dev)

            # 输入清洗
            x_pred = torch.nan_to_num(x_pred.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            x_gt   = torch.nan_to_num(x_gt.float(),   nan=0.0, posinf=1e4, neginf=-1e4)

            spec_p = torch.stft(x_pred, n_fft=n_fft_eff, hop_length=hop_eff,
                                win_length=win_eff, window=win, center=use_center,
                                return_complex=True)
            spec_g = torch.stft(x_gt,   n_fft=n_fft_eff, hop_length=hop_eff,
                                win_length=win_eff, window=win, center=use_center,
                                return_complex=True)

            mag_p = _mag(spec_p)  # [B*, F, T']
            mag_g = _mag(spec_g)

            # 每个 band 的 log1p 能量
            Eb_p = torch.log1p(self._band_energy(mag_p))  # [B*, nb]
            Eb_g = torch.log1p(self._band_energy(mag_g))  # [B*, nb]

            # 自适应高频加权
            w = self.base_w.to(dev).expand(Eb_p.size(0), -1)  # [B*, nb]
            if self.adaptive:
                need = (Eb_g > Eb_p).float()                  # 需要回补的 band
                w = w * (1.0 + 0.2 * need)                    # ×1.2（上限由 EMA 护栏限制）

            # EMA + 护栏
            new_w = _safe_weight_ema(w, self._ewma_w, ema=self.ema, wmin=self.wmin, wmax=self.wmax)
            self._ewma_w = new_w.detach()

            # 逐样本归一化（防止分母为 0）
            denom  = new_w.sum(dim=-1, keepdim=True) + 1e-6
            w_norm = new_w / denom

            # band-wise L1
            per_band = torch.abs(Eb_p - Eb_g)                 # [B*, nb]
            loss = (w_norm * per_band).sum(dim=-1).mean()     # 标量

            # warmup
            if self.warmup_steps > 0:
                self._steps += 1
                scale = min(1.0, float(self._steps.item()) / float(self.warmup_steps))
                loss = loss * scale

        # 兜底：若出现异常，返回 0（上层已做 warn）
        if not torch.isfinite(loss):
            return torch.zeros([], device=x_pred.device)

        return self.weight * loss
