import numpy as np
import torch


class DCAWController:
    """
    Dual-Channel Adaptive Window controller.

    Computes per-timestep window indicator W(t) and smoothing weights S(t)
    from audio prosody intensity and motion intensity.

    Prosody is estimated from short-term energy/variance of DeepSpeech features.
    Motion intensity is estimated from pose/exp deltas if available.
    """

    def __init__(self,
                 min_window: int = 96,
                 max_window: int = 160,
                 audio_weight: float = 0.6,
                 motion_weight: float = 0.4,
                 eps: float = 1e-6,
                 use_smooth_window: bool = True,
                 ema_alpha: float = 0.3,
                 hysteresis_threshold: float = 0.05,
                 au45_fixed_window: int = 64,
                 high_freq_exemption_lambda: float = 0.2):
        self.min_window = min_window
        self.max_window = max_window
        self.audio_weight = audio_weight
        self.motion_weight = motion_weight
        self.eps = eps
        self.use_smooth_window = use_smooth_window
        self.ema_alpha = ema_alpha  # EMA平滑系数
        self.hysteresis_threshold = hysteresis_threshold  # 滞回阈值
        self.au45_fixed_window = au45_fixed_window  # AU45固定小窗
        self.high_freq_exemption_lambda = high_freq_exemption_lambda  # 高频豁免系数
        self._prev_window_state = None  # 用于滞回控制
        self._prev_W_discrete = None  # 用于滞回控制的离散窗口状态

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _normalize(self, v):
        v = np.asarray(v, dtype=np.float32)
        v_min = np.min(v)
        v_max = np.max(v)
        if v_max - v_min < self.eps:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)

    def _prosody_intensity(self, audio_window):
        """
        audio_window: [T, A, F] DeepSpeech stacked features.
        Returns per-timestep intensity in [0, 1].
        """
        a = self._to_numpy(audio_window)
        # a is expected as [T, W, C] or [T, C] where C are DeepSpeech logits per class
        def softmax(x, axis=-1):
            x = x - np.max(x, axis=axis, keepdims=True)
            e = np.exp(x)
            return e / (np.sum(e, axis=axis, keepdims=True) + self.eps)

        if a.ndim == 3:
            # use center frame of each local audio window
            center = a.shape[1] // 2
            logits = a[:, center, :]
        elif a.ndim == 2:
            logits = a
        else:
            # fallback: flatten last
            logits = a.reshape(a.shape[0], -1)

        probs = softmax(logits, axis=1)
        dt = np.max(probs, axis=1)  # Dt = max_i P_ti
        # Smooth with W=32
        W = 32
        pad = (W - 1) // 2
        dt_pad = np.pad(dt, (pad, pad), mode='edge')
        smoothed = np.convolve(dt_pad, np.ones(W) / W, mode='valid')
        return self._normalize(smoothed)

    def _motion_intensity(self, params_window):
        """
        params_window: [T, P] with pose(0:6), AU45 at index 6, and expr 7:.
        Returns per-timestep motion intensity in [0, 1].
        """
        p = self._to_numpy(params_window)
        if p is None:
            return None
        if p.shape[0] < 2:
            return np.zeros((p.shape[0],), dtype=np.float32)
        dp = np.diff(p, axis=0)
        # Pose + AU + expr deltas
        weights = np.ones(p.shape[1], dtype=np.float32)
        # Emphasize pose motion a bit more
        weights[:6] = 2.0
        # AU channel (index 6)
        if p.shape[1] > 6:
            weights[6] = 1.5
        # Weighted L2 per step, then pad to length T
        step_energy = np.sqrt(np.sum((dp * weights[None, :]) ** 2, axis=1) + self.eps)
        # pad到左首，truncate到T
        step_energy = np.concatenate([[step_energy[0]], step_energy], axis=0)
        if step_energy.shape[0] > p.shape[0]:
            step_energy = step_energy[:p.shape[0]]
        elif step_energy.shape[0] < p.shape[0]:
            step_energy = np.pad(step_energy, (0, p.shape[0]-step_energy.shape[0]), mode='edge')
        # Smooth
        k = 5
        pad = (k - 1) // 2
        e_pad = np.pad(step_energy, (pad, pad), mode='edge')
        smoothed = np.convolve(e_pad, np.ones(k) / k, mode='valid')
        return self._normalize(smoothed)

    def _smooth_window_transition(self, dt, use_hysteresis=True):
        """
        平滑窗口切换：使用滞回+EMA平滑或连续sigmoid形式
        
        Args:
            dt: 韵律强度 [T]
            use_hysteresis: 是否使用滞回控制
            
        Returns:
            W_t: 平滑后的窗口大小 [T]
        """
        T = len(dt)
        
        if not self.use_smooth_window:
            # 原始硬切换
            W_t = np.full_like(dt, 128, dtype=np.int32)
            W_t[dt > 0.6] = 64
            W_t[dt <= 0.1] = 256
            return W_t
        
        # 方案1: 连续sigmoid形式的软窗分配
        # 将dt映射到连续窗口大小，然后离散化
        # 使用sigmoid实现平滑过渡
        def sigmoid_window(x, center=0.35, scale=10.0):
            """将dt映射到窗口大小，使用sigmoid平滑过渡"""
            # 归一化到[-1, 1]范围，然后映射到窗口大小
            # 高dt -> 小窗口(64), 低dt -> 大窗口(256)
            x_norm = (x - center) * scale
            sig_val = 1.0 / (1.0 + np.exp(-x_norm))
            # 映射到窗口大小: 64 <-> 256
            window_size = 64 + (256 - 64) * (1 - sig_val)
            return window_size
        
        # 计算连续窗口大小
        W_continuous = sigmoid_window(dt)
        
        # 应用EMA平滑（如果使用滞回）
        if use_hysteresis and self._prev_window_state is not None:
            # EMA平滑: W_t = EMA(W_{t-1}, target)
            prev_state = self._prev_window_state
            if len(prev_state) == T:
                W_continuous = self.ema_alpha * W_continuous + (1 - self.ema_alpha) * prev_state
            self._prev_window_state = W_continuous.copy()
        else:
            self._prev_window_state = W_continuous.copy()
        
        # 离散化到最近的窗口大小
        W_t = np.zeros_like(W_continuous, dtype=np.int32)
        for i, w in enumerate(W_continuous):
            if w < 96:
                W_t[i] = 64
            elif w < 192:
                W_t[i] = 128
            else:
                W_t[i] = 256
        
        # 滞回控制：避免频繁切换
        if use_hysteresis and len(self._prev_window_state) == T:
            prev_W = np.zeros_like(W_t, dtype=np.int32)
            prev_W[:] = 128  # 默认值
            if hasattr(self, '_prev_W_discrete') and self._prev_W_discrete is not None and len(self._prev_W_discrete) == T:
                prev_W = self._prev_W_discrete.copy()
            
            for i in range(T):
                if prev_W[i] == 64:
                    # 从64切换到128需要dt降到0.6 - hysteresis_threshold
                    if dt[i] <= 0.6 - self.hysteresis_threshold:
                        W_t[i] = 128
                    elif dt[i] <= 0.1 - self.hysteresis_threshold:
                        W_t[i] = 256
                elif prev_W[i] == 128:
                    # 从128切换到64需要dt > 0.6 + hysteresis_threshold
                    if dt[i] > 0.6 + self.hysteresis_threshold:
                        W_t[i] = 64
                    # 从128切换到256需要dt <= 0.1 - hysteresis_threshold
                    elif dt[i] <= 0.1 - self.hysteresis_threshold:
                        W_t[i] = 256
                elif prev_W[i] == 256:
                    # 从256切换到128需要dt > 0.1 + hysteresis_threshold
                    if dt[i] > 0.1 + self.hysteresis_threshold and dt[i] <= 0.6:
                        W_t[i] = 128
                    elif dt[i] > 0.6 + self.hysteresis_threshold:
                        W_t[i] = 64
            
            self._prev_W_discrete = W_t.copy()
        
        return W_t

    def _compute_high_freq_exemption(self, params_window):
        """
        计算高频特征通道豁免项 H_f(t)
        用于在S(t)中减轻对高频动作（如AU45）的抑制
        """
        if params_window is None:
            return None
        
        p = self._to_numpy(params_window)
        if p is None or p.shape[0] < 2:
            return None
        
        # 计算AU45的瞬时变化率（高频特征）
        if p.shape[1] > 6:
            au45 = p[:, 6:7]
            au45_diff = np.abs(np.diff(au45, axis=0))
            # 归一化高频能量
            h_f = np.concatenate([[au45_diff[0]], au45_diff], axis=0)
            if h_f.shape[0] > p.shape[0]:
                h_f = h_f[:p.shape[0]]
            elif h_f.shape[0] < p.shape[0]:
                h_f = np.pad(h_f, (0, p.shape[0] - h_f.shape[0]), mode='edge')
            # 归一化到[0, 1]
            h_f = self._normalize(h_f)
            return h_f
        return None

    def compute(self, audio_window, params_window=None):
        """
        Compute W(t) and S(t) from given windowed audio and optional params.

        Returns:
        - W_t: int window suggestion per step in [min_window, max_window]
        - S_t: smoothing weights in [0.5, 1.5], same length as T, higher -> stronger loss
        """
        prosody = self._prosody_intensity(audio_window)
        motion = self._motion_intensity(params_window) if params_window is not None else None
        # --- robust length match ---
        len_p = len(prosody)
        len_m = len(motion) if motion is not None else len_p
        min_len = min(len_p, len_m)
        prosody = prosody[:min_len]
        if motion is not None:
            motion = motion[:min_len]
        # --- end ---
        if motion is None:
            fused = prosody
        else:
            fused = self.audio_weight * prosody + self.motion_weight * motion
            fused = self._normalize(fused)

        # 使用平滑窗口切换替代硬切换
        dt = self._prosody_intensity(audio_window)
        W_t = self._smooth_window_transition(dt, use_hysteresis=True)
        
        # 对AU45通道使用固定小窗（如果启用）
        # 注意：这里我们在窗口选择层面处理，实际在损失中会进一步处理

        # Smoothing weight: combine fused intensity with window mask emphasis later
        S_t = 0.5 + 1.0 * fused  # [0.5, 1.5]
        
        # 高频特征通道豁免项：S'(t) = S(t) - λ·H_f(t)
        h_f = self._compute_high_freq_exemption(params_window)
        if h_f is not None and len(h_f) == len(S_t):
            S_t = S_t - self.high_freq_exemption_lambda * h_f
            # 确保S_t仍在有效范围内
            S_t = np.clip(S_t, 0.5, 1.5)
        
        return W_t, S_t

    def make_window_mask(self, T: int, W_sel: int):
        """Return a per-timestep mask emphasizing the central W_sel region within T.
        Center the mask; values 1.0 inside, 0.5 outside to de-emphasize.
        """
        mask = np.full((T,), 0.5, dtype=np.float32)
        if W_sel >= T:
            mask[:] = 1.0
            return mask
        start = (T - W_sel) // 2
        end = start + W_sel
        mask[start:end] = 1.0
        return mask
    
    # audio2face/modules/dcaw_controller.py

    # modules/dcaw_controller.py （插入）

    def make_au45_narrow_window(T, center_idx=None, half_width=8, temperature=0.75, device=None):
        """
        AU45 专用窄软窗（不乘 S(t)），默认 ±8 帧，高斯核 + 温度，避免 0 权重。
        返回 shape: [T] 的 float 权重。
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        idx  = torch.arange(T, device=device)
        c    = (center_idx if center_idx is not None else T//2)
        c    = torch.tensor(c, device=device)
        dist = (idx - c).float().abs()
        w    = torch.exp(-(dist ** 2) / (2.0 * (half_width ** 2) * temperature))
        w    = torch.clamp(w, min=1e-3)
        return (w / (w.max() + 1e-6))



    def suggest_stride(self, audio_window):
        """
        For inference without params, suggest step size based on audio only.
        Returns a single stride integer derived from median W(t).
        """
        W_t, _ = self.compute(audio_window, None)
        mW = int(np.median(W_t))
        # Map to strides roughly half of window size, clamped
        stride = 64 if mW <= 64 else (96 if mW <= 128 else 127)
        return stride


