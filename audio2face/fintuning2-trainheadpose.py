# -*- coding: utf-8 -*-
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import TfaceGAN, NLayerDiscriminator
from dataset102 import Facial_Dataset

# 仅保留本脚本需要的增强损失
from modules.enhanced_losses import (
    AU45BranchLoss,            # AU45 专支路（含峰值保持）
    ExpressionEnergyBandLoss,  # 分带表情能量 + 泄压阀
    PoseHuberLoss,             # 姿态 Huber 正则
    BandedFeatureMatchingLoss  # 分带 FM（可选）
)

# 使用“稳定版” STFT
from losses.stft_stable import UltraShortWindowSTFTLossStable, BandedSTFTLossStable

# AU45 局部判别器
from modules.au45_discriminator import AU45LocalDiscriminator

import argparse, os
import numpy as np

# ----------------------------
# argparse
# ----------------------------
parser = argparse.ArgumentParser(description='Train_setting')
parser.add_argument('--audiopath', type=str, default='/root/FACIAL-main/examples/audio_preprocessed/final_audio.pkl')
parser.add_argument('--npzpath', type=str,   default='/root/FACIAL-main/video_preprocess/train1.npz')
parser.add_argument('--cvspath', type=str,   default='/root/FACIAL-main/video_preprocess/openface/output.csv')
parser.add_argument('--pretainpath_gen', type=str, default='/root/FACIAL-main/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl')
parser.add_argument('--savepath', type=str,  default='/root/FACIAL-main/audio2face/checkpoint/full')

# DCAW / 平滑（控制层）
parser.add_argument('--use_dcaw', action='store_true', help='Enable dual-channel adaptive window and smoothing weights')
parser.add_argument('--soft_gate', action='store_true', help='Enable differentiable soft gating for attention windowing')
parser.add_argument('--soft_wmin', type=int, default=64,  help='Soft gating minimum window radius baseline')
parser.add_argument('--soft_wmax', type=int, default=256, help='Soft gating maximum window radius baseline')

# 正则与权重（损失层）
parser.add_argument('--tv_weight',       type=float, default=0.1,   help='Temporal TV loss base weight on outputs')
parser.add_argument('--pose_tv_extra_weight', type=float, default=1.5, help='Extra TV multiplier on pose (first 6 dims)')
parser.add_argument('--feat_tv_weight',  type=float, default=0.05,  help='Temporal TV loss weight on attention features')
parser.add_argument('--jerk_weight',     type=float, default=0.02,  help='Jerk (3rd-order) penalty weight on pose')
parser.add_argument('--expr_energy_weight', type=float, default=0.5, help='Expression energy compensation loss weight (β)')

# 判别器 stride
parser.add_argument('--disc_stride_sync',   action='store_true', help='Enable synchronized stride sampling for discriminator')
parser.add_argument('--disc_random_stride', action='store_true', help='Enable random stride training for discriminator to reduce domain shift')

# FM / STFT（频域层）
parser.add_argument('--feat_match_weight', type=float, default=6.5, help='Feature matching loss weight')
parser.add_argument('--use_banded_fm', action='store_true', help='Enable banded feature matching loss')

parser.add_argument('--use_banded_stft', action='store_true', help='Enable banded STFT loss (stable)')
parser.add_argument('--stft_weight',     type=float, default=0.18, help='(stable) Banded STFT total weight; internally warmup')
parser.add_argument('--stft_weight_pose',type=float, default=0.0,  help='Optional tiny STFT weight on pose (safe start 0.0)')

parser.add_argument('--use_ultra_short_stft', action='store_true', help='Enable ultra short window STFT loss for AU45 (stable)')
parser.add_argument('--ultra_stft_weight',    type=float, default=0.0,  help='Ultra-short STFT weight for AU45 (start 0.0, then 0.05~0.10)')

# AU45 专支路 + 局部判别器
parser.add_argument('--use_au45_branch', action='store_true', help='Enable AU45 dedicated branch and local discriminator')
parser.add_argument('--au45_disc_weight', type=float, default=0.35, help='AU45 local discriminator loss weight')

# 姿态正则/窗口切换抑制（控制层）
parser.add_argument('--use_huber_pose', action='store_true', help='Enable Huber loss for pose with acceleration regularization')
parser.add_argument('--max_window_change', type=int, default=20, help='Max single-step window radius change')
parser.add_argument('--radius_change_weight', type=float, default=0.3, help='L1 on second diff of window radius')
parser.add_argument('--min_dwell_frames', type=int, default=18, help='Minimum dwell frames before switching radius')

# 分带表情能量（可选）
parser.add_argument('--use_banded_expr', action='store_true', help='Enable banded expression energy loss with pressure valve')

opt = parser.parse_args()
os.makedirs(opt.savepath, exist_ok=True)

# ----------------------------
# dataset & loader
# ----------------------------
audio_paths = [opt.audiopath]
npz_paths   = [opt.npzpath]
cvs_paths   = [opt.cvspath]

batchsz = 16
epochs  = 11
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)

training_set = Facial_Dataset(audio_paths, npz_paths, cvs_paths, use_dcaw=opt.use_dcaw)
train_loader = DataLoader(training_set, batch_size=batchsz, shuffle=True, drop_last=True, pin_memory=True)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for p in net.parameters():
                p.requires_grad = requires_grad

# ———— helpers ————
def weighted_mse(a, b, w):
    # a, b: [B, T, C] or [B, T]; w: [B, T]
    min_t = min(a.shape[1], b.shape[1], w.shape[1])
    a, b, w = a[:, :min_t, ...], b[:, :min_t, ...], w[:, :min_t]
    se = (a - b) ** 2
    w3 = w.unsqueeze(-1) if se.dim() > 2 else w
    return torch.mean(se * w3)

def temporal_tv(a, w):
    """
    Robust TV: automatically aligns time length of a and w.
    a: [B, T, C], w: [B, T]
    """
    T = min(a.shape[1], w.shape[1])
    if T <= 1:
        return torch.tensor(0., device=a.device)
    a_use = a[:, :T, :]
    w_use = w[:, :T]
    diff = torch.abs(a_use[:, 1:, :] - a_use[:, :-1, :])  # [B, T-1, C]
    w_next = w_use[:, 1:]                                 # [B, T-1]
    return torch.mean(diff * w_next.unsqueeze(-1))

def expression_energy_loss(pred, gt, w=None):
    # L_expr = (E_gt - E_pred)^2；表情通道从 idx 7 开始
    pred_expr = pred[:, :, 7:] if pred.shape[2] > 7 else pred
    gt_expr   = gt[:, :, 7:] if   gt.shape[2] > 7 else gt
    E_pred = torch.norm(pred_expr, p=2, dim=2)  # [B, T]
    E_gt   = torch.norm(gt_expr,   p=2, dim=2)
    diff   = (E_gt - E_pred) ** 2
    if w is not None:
        min_t = min(diff.shape[1], w.shape[1])
        diff  = diff[:, :min_t] * w[:, :min_t]
    return diff.mean()

# jerk（第三阶）惩罚，仅作用在 pose 前6维 —— 修正版
def jerk_loss_pose(seq_bt6, w=None):
    """
    seq_bt6: [B, T, 6] pose
    先算加速度 a_t = y_{t+1}-2*y_t+y_{t-1} （长度 T-2）
    再算 jerk = |a_{t} - a_{t-1}| （长度 T-3）
    与权重 w（长度 T）对齐时，从 t=2 开始对齐到 jerk 的时间轴。
    """
    B, T, C = seq_bt6.shape
    if T < 4:
        return torch.tensor(0., device=seq_bt6.device)

    y = seq_bt6
    acc = y[:, 2:, :] - 2*y[:, 1:-1, :] + y[:, :-2, :]     # [B, T-2, 6]
    j   = torch.abs(acc[:, 1:, :] - acc[:, :-1, :])        # [B, T-3, 6]

    if w is not None:
        # w 是长度 T 的时间权重；jerk 从 t=2 开始（对应原序列索引 2..T-2）
        w_use = w[:, 2:]                                    # [B, T-2]
        Tw   = min(w_use.shape[1]-1, j.shape[1])            # -1 后与 j 的 T-3 对齐
        if Tw > 0:
            jw = w_use[:, 1:1+Tw].unsqueeze(-1)             # [B, Tw, 1]
            j  = j[:, :Tw, :] * jw
        else:
            j = torch.zeros([], device=seq_bt6.device)
    return j.mean() if j.numel() > 0 else torch.tensor(0., device=seq_bt6.device)

# 后备：若 controller 未提供 AU45 窄窗接口，则在本文件生成一个
def make_au45_narrow_window_local(T, center_idx=None, half_width=8, temperature=0.75, device=device):
    idx  = torch.arange(T, device=device)
    c    = (center_idx if center_idx is not None else T//2)
    c    = torch.tensor(c, device=device)
    dist = (idx - c).float().abs()
    w    = torch.exp(-(dist ** 2) / (2.0 * (half_width ** 2) * temperature))
    w    = torch.clamp(w, min=1e-3)
    return (w / (w.max() + 1e-6))  # [T]

# 供可视化/调试的半径二阶正则（需要 m 是随时间的窗掩码或其近似）
def compute_window_change(m):
    return torch.abs(m[:, 1:] - m[:, :-1]) if m.shape[1] >= 2 else None

def main():
    lr = 1e-4
    modelgen = TfaceGAN().to(device)
    modeldis = NLayerDiscriminator().to(device)

    # AU45 local D（可选）
    modeldis_au45 = AU45LocalDiscriminator().to(device) if opt.use_au45_branch else None

    # 预加载生成器
    try:
        state = torch.load(opt.pretainpath_gen, map_location='cpu')
        modelgen.load_state_dict(state, strict=False)
    except Exception as e:
        print('[warn] load_state_dict(strict=False):', e)
    print(modelgen); print(modeldis); print(modeldis_au45 if modeldis_au45 else 'No AU45 D')

    # optim
    optimG = optim.Adam(modelgen.parameters(), lr=lr*0.1)
    optimD = optim.Adam(modeldis.parameters(), lr=lr*0.1)
    optimD_au45 = optim.Adam(modeldis_au45.parameters(), lr=lr*0.1) if modeldis_au45 is not None else None

    # 基础 loss
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    # AU45 分支（含峰值保持）
    au45_branch_loss = None
    if opt.use_au45_branch:
        au45_branch_loss = AU45BranchLoss(
            base_weight=0.5, high_freq_weight=2.0, temporal_weight=1.0,
            peak_weight=2.0, peak_window=10
        )

    # —— 稳定版 STFT（不使用自适应与 EMA，避免 1024 vs 96 形状冲突） ——
    banded_stft_loss = BandedSTFTLossStable(
        n_fft=128, hop_length=32, win_length=96,
        bands=((0,64),(64,96),(96,128)),
        band_weights=(0.05, 0.05, 0.10),
        adaptive=False, ema=0.0, wmin=0.5, wmax=1.2,
        weight=opt.stft_weight, warmup_steps=0
    )

    # （可选）分带 FM
    banded_fm_loss = BandedFeatureMatchingLoss(weights=[0.5, 1.0, 1.5]) if opt.use_banded_fm else None

    # （可选）分带表情能量
    banded_expr_loss = ExpressionEnergyBandLoss() if opt.use_banded_expr else None

    # （可选）姿态 Huber
    huber_pose_loss = PoseHuberLoss(
        delta=0.75,  # 更快从 L2 过渡到 L1，抑制硬切
        max_window_change=opt.max_window_change,
        min_dwell_frames=opt.min_dwell_frames,
        radius_change_weight=opt.radius_change_weight
    ) if opt.use_huber_pose else None

    # —— 超短窗 STFT，用于 AU45 —— #
    if opt.use_ultra_short_stft:
        ultra_stft_loss = BandedSTFTLossStable(
            n_fft=32, hop_length=8, win_length=24,
            bands=((0,16), (16,24), (24,32)),
            band_weights=(0.25, 0.25, 0.50),
            adaptive=False, ema=0.0, wmin=0.5, wmax=1.2,
            weight=opt.ultra_stft_weight, warmup_steps=0
        )
    else:
        ultra_stft_loss = None

    # ———— train ————
    for epoch in range(epochs):
        if epoch % 5 == 0:
            torch.save(modelgen.state_dict(), os.path.join(opt.savepath, f'Gen-{epoch}.mdl'))
            torch.save(modeldis.state_dict(), os.path.join(opt.savepath, f'Dis-{epoch}.mdl'))

        for step, batch in enumerate(train_loader):
            if opt.use_dcaw:
                x, y, s, m = batch
                # 对齐 s,m 时间长度
                min_len = min(s.shape[1], m.shape[1])
                s = s[:, :min_len]; m = m[:, :min_len]
            else:
                x, y = batch
                s = torch.ones((x.shape[0], x.shape[1]), dtype=torch.float32)
                m = torch.ones_like(s)

            modelgen.train()
            x, y = x.to(device), y.to(device)
            s, m = s.to(device), m.to(device)
            motiony = y[:,1:,:] - y[:,:-1,:]

            # 判别器开梯度
            set_requires_grad(modeldis, True)
            if modeldis_au45 is not None:
                set_requires_grad(modeldis_au45, True)

            # stride 采样（判别器）
            def apply_stride_sampling(data, stride_range=(1, 4)):
                B, T, C = data.shape
                if opt.disc_random_stride:
                    stride  = np.random.randint(stride_range[0], stride_range[1] + 1)
                    indices = torch.arange(0, T, stride, device=data.device)
                    sampled = data[:, indices, :]
                    mask    = torch.zeros((B, T), device=data.device)
                    mask[:, indices] = 1.0
                    return sampled, mask
                elif opt.disc_stride_sync and opt.use_dcaw:
                    stride  = 2
                    indices = torch.arange(0, T, stride, device=data.device)
                    sampled = data[:, indices, :]
                    mask    = torch.zeros((B, T), device=data.device)
                    mask[:, indices] = 1.0
                    return sampled, mask
                else:
                    return data, torch.ones((B, T), device=data.device)

            # =========================
            # 判别器：Real
            # =========================
            real_data = torch.cat([y, motiony], dim=1)   # [B, 2T-1, C]
            real_samp, real_mask = apply_stride_sampling(real_data)
            real_tmask = m if not opt.disc_random_stride else real_mask

            # 可选 FM 特征
            if hasattr(modeldis, 'extract_features'):
                real_feats = modeldis.extract_features(real_data)
            else:
                real_feats = None

            predr = modeldis(real_samp, tmask=real_tmask)
            lossr = mse(torch.ones_like(predr), predr)

            # AU45 D（real）
            lossr_au45 = 0.0
            if modeldis_au45 is not None:
                real_au45 = y[:, :, 6:7]
                real_au45_samp = real_au45[:, :real_samp.shape[1], :]
                if real_au45_samp.numel() > 0:
                    predr_au = modeldis_au45(real_au45_samp.squeeze(-1), tmask=real_tmask)
                    lossr_au45 = mse(torch.ones_like(predr_au), predr_au)

            # =========================
            # 判别器：Fake
            # =========================
            if opt.use_dcaw:
                yf, feats = modelgen(
                    x, y[:,:1,:],
                    attn_mask=m, return_feats=True,
                    soft_gate=opt.soft_gate, soft_wmin=opt.soft_wmin, soft_wmax=opt.soft_wmax
                )
            else:
                yf = modelgen(x, y[:,:1,:]); feats = None

            motionlogits = yf[:,1:,:] - yf[:,:-1,:]

            # —— 速度一致性（仅作用于 pose 前6维） ——
            vel_cons = torch.tensor(0., device=device)
            if yf.shape[2] >= 6:
                vel_pred = yf[:, 1:, :6] - yf[:, :-1, :6]
                vel_gt   = y[:,  1:, :6] - y[:,  :-1, :6]
                if opt.use_dcaw:
                    min_tv = min(vel_pred.shape[1], s.shape[1]-1, m.shape[1]-1)
                    sw_v = (s[:, 1:1+min_tv] * m[:, 1:1+min_tv]).to(device)
                    vel_cons = 0.15 * weighted_mse(vel_pred[:, :min_tv, :], vel_gt[:, :min_tv, :], sw_v)
                else:
                    vel_cons = 0.15 * nn.MSELoss()(vel_pred, vel_gt)

            fake_data = torch.cat([yf, motionlogits], dim=1).detach()  # 给 D 用，需 detach
            fake_samp, fake_mask = apply_stride_sampling(fake_data)
            fake_tmask = m if not opt.disc_random_stride else fake_mask

            predf = modeldis(fake_samp, tmask=fake_tmask)
            lossf = mse(torch.zeros_like(predf), predf)

            # AU45 D（fake）
            lossf_au45 = 0.0
            if modeldis_au45 is not None:
                fake_au45 = yf[:, :, 6:7].detach()
                fake_au45_samp = fake_au45[:, :fake_samp.shape[1], :]
                if fake_au45_samp.numel() > 0:
                    predf_au = modeldis_au45(fake_au45_samp.squeeze(-1), tmask=fake_tmask)
                    lossf_au45 = mse(torch.zeros_like(predf_au), predf_au)

            # FM（若支持）
            loss_feat_match = 0.0
            if real_feats is not None and hasattr(modeldis, 'extract_features'):
                fake_feats = modeldis.extract_features(torch.cat([yf, motionlogits], dim=1).detach())
                if opt.use_banded_fm and banded_fm_loss is not None:
                    loss_feat_match = banded_fm_loss(real_feats, fake_feats)
                else:
                    if isinstance(real_feats, (list, tuple)) and isinstance(fake_feats, (list, tuple)):
                        for rf, ff in zip(real_feats, fake_feats):
                            loss_feat_match += l1(rf, ff)
                    else:
                        loss_feat_match = l1(real_feats, fake_feats)

            lossD = lossr + lossf + opt.feat_match_weight * loss_feat_match
            if modeldis_au45 is not None:
                lossD += opt.au45_disc_weight * (lossr_au45 + lossf_au45)

            optimD.zero_grad()
            if optimD_au45 is not None: optimD_au45.zero_grad()
            lossD.backward()
            optimD.step()
            if optimD_au45 is not None: optimD_au45.step()

            # --------------------
            # Generator（不 detach）
            # --------------------
            set_requires_grad(modeldis, False)
            if modeldis_au45 is not None:
                set_requires_grad(modeldis_au45, False)

            # 单帧重建
            loss_s = 10*( l1(yf[:,:1,:6], y[:,:1,:6]) + l1(yf[:,:1,6], y[:,:1,6]) + l1(yf[:,:1,6:], y[:,:1,6:]) )

            # sw（加权）
            if opt.use_dcaw:
                min_t = min(s.shape[1], m.shape[1], yf.shape[1], y.shape[1])
                s_use, m_use = s[:, :min_t], m[:, :min_t]
                sw    = s_use * m_use
                sw_n  = sw[:, 1:]
                lossg_e  = 20*weighted_mse(yf[:, :min_t, 7:], y[:, :min_t, 7:], sw)
                lossg_em = 200*weighted_mse((yf[:,1:min_t,:]-yf[:,:min_t-1,:])[:,:,7:],
                                            (y[:,1:min_t,:]-y[:,:min_t-1,:])[:,:,7:], sw_n)
            else:
                sw = None
                lossg_e  = 20*nn.MSELoss()(yf[:,:,7:], y[:,:,7:])
                lossg_em = 200*nn.MSELoss()(motionlogits[:,:,7:], motiony[:,:,7:])

            # AU45（完全不乘 S(t)，仅窄软窗）
            loss_au  = torch.tensor(0., device=device)
            loss_aum = torch.tensor(0., device=device)
            loss_au_stft_val = torch.tensor(0., device=device)
            if opt.use_au45_branch and au45_branch_loss is not None:
                T_use = min(yf.shape[1], y.shape[1])
                try:
                    from modules.dcaw_controller import make_au45_narrow_window
                    narrow_w = make_au45_narrow_window(T_use, center_idx=None, half_width=8, temperature=0.75)
                    if isinstance(narrow_w, np.ndarray):
                        narrow_w = torch.from_numpy(narrow_w).float().to(device)
                except Exception:
                    narrow_w = make_au45_narrow_window_local(T_use)  # [T]
                sw_au = narrow_w.unsqueeze(0).expand(yf.shape[0], -1)  # [B, T]

                loss_au  = au45_branch_loss(yf[:, :T_use, 6], y[:, :T_use, 6], sw_au)
                if T_use > 1:
                    diff_pred = yf[:, 1:T_use, 6] - yf[:, :T_use-1, 6]
                    diff_gt   =  y[:, 1:T_use, 6] -  y[:, :T_use-1, 6]
                    loss_aum  = au45_branch_loss(diff_pred, diff_gt, sw_au[:, 1:T_use])

                if opt.use_ultra_short_stft and ultra_stft_loss is not None and ultra_stft_loss.weight > 0:
                    try:
                        # 这里 ultra_stft_loss 支持 [B, T] 输入
                        au45_pred = yf[:, :T_use, 6]
                        au45_gt   =  y[:, :T_use, 6]
                        loss_au_stft_val = opt.ultra_stft_weight * ultra_stft_loss(au45_pred, au45_gt)
                        if torch.isnan(loss_au_stft_val) or torch.isinf(loss_au_stft_val):
                            print('[warn] ultra STFT loss NaN/Inf -> 0'); loss_au_stft_val = torch.zeros([], device=device)
                    except Exception as e:
                        print('[warn] ultra STFT failed:', e); loss_au_stft_val = torch.zeros([], device=device)

            # —— 生成器对抗损失（全局判别器） ——
            def resample_for_g(data_3d):
                samp, mask = apply_stride_sampling(data_3d, stride_range=(1, 4))
                tmask = m if not opt.disc_random_stride else mask
                return samp, tmask

            gen_cat = torch.cat([yf, motionlogits], dim=1)              # [B, 2T-1, C]
            gen_samp, gen_tmask = resample_for_g(gen_cat)
            pred_g = modeldis(gen_samp, tmask=gen_tmask)
            lossg_gan = mse(torch.ones_like(pred_g), pred_g)

            # —— 生成器 AU45 对抗损失（可选） ——
            if opt.use_au45_branch and modeldis_au45 is not None:
                au45_seq = yf[:, :, 6:7]                                # [B, T, 1]
                au45_samp = au45_seq[:, :gen_samp.shape[1], :].squeeze(-1)  # [B, T]
                pred_g_au = modeldis_au45(au45_samp, tmask=gen_tmask)
                lossg_gan_au45 = mse(torch.ones_like(pred_g_au), pred_g_au)
            else:
                lossg_gan_au45 = torch.tensor(0., device=device)

            # —— 输出 TV：pose 通道额外加权 ——
            # 全通道 TV
            if sw is not None:
                loss_tv_all = temporal_tv(yf, sw)
            else:
                diff_all = torch.abs(yf[:,1:,:] - yf[:,:-1,:])
                loss_tv_all = diff_all.mean()
            # pose 专属 TV（在前6维上叠加一个额外倍数）
            if yf.shape[2] >= 6:
                if sw is not None:
                    loss_tv_pose = temporal_tv(yf[:,:,:6], sw) * opt.pose_tv_extra_weight
                else:
                    diff_pose = torch.abs(yf[:,1:,:6] - yf[:,:-1,:6]).mean() * opt.pose_tv_extra_weight
                    loss_tv_pose = diff_pose
            else:
                loss_tv_pose = torch.tensor(0., device=device)
            loss_tv = opt.tv_weight * (loss_tv_all + loss_tv_pose)

            # —— 特征 TV（若生成器返回中间特征） ——
            loss_feat_tv = torch.tensor(0., device=device)
            if feats is not None and isinstance(feats, (list, tuple)):
                for f in feats:
                    if f is None: continue
                    if f.dim() == 3 and f.shape[1] > 1:
                        if sw is not None:
                            loss_feat_tv += temporal_tv(f, sw)
                        else:
                            loss_feat_tv += torch.mean(torch.abs(f[:, 1:, :] - f[:, :-1, :]))
                loss_feat_tv = opt.feat_tv_weight * loss_feat_tv

            # —— 表情能量一致性 ——
            if sw is not None:
                loss_expr_energy = opt.expr_energy_weight * expression_energy_loss(yf, y, sw)
            else:
                loss_expr_energy = opt.expr_energy_weight * expression_energy_loss(yf, y, None)

            # —— Banded STFT（频域层） ——
            loss_stft = torch.tensor(0., device=device)
            if opt.use_banded_stft and banded_stft_loss.weight > 0:
                try:
                    # 表情通道（7:），转成 [B, C, T]
                    expr_pred = yf[:, :, 7:].transpose(1, 2).contiguous()
                    expr_gt   =  y[:, :, 7:].transpose(1, 2).contiguous()
                    loss_stft_expr = banded_stft_loss(expr_pred, expr_gt)

                    # 可选：对姿态施加很小的频域约束
                    if opt.stft_weight_pose > 0 and yf.shape[2] >= 6:
                        pose_pred = yf[:, :, :6].transpose(1, 2).contiguous()
                        pose_gt   =  y[:, :, :6].transpose(1, 2).contiguous()
                        loss_stft_pose = banded_stft_loss(pose_pred, pose_gt)
                        loss_stft = loss_stft_expr + loss_stft_pose
                    else:
                        loss_stft = loss_stft_expr
                except Exception as e:
                    print('[warn] banded STFT failed:', e)
                    loss_stft = torch.zeros([], device=device)

            # —— 姿态 Huber（可选） ——
            loss_pose  = torch.tensor(0., device=device)
            loss_posem = torch.tensor(0., device=device)
            if opt.use_huber_pose and huber_pose_loss is not None:
                T_pose = min(yf.shape[1], y.shape[1])
                if sw is not None:
                    sw_pose = sw[:, :T_pose]
                    wchg = compute_window_change(m[:, :T_pose])
                else:
                    sw_pose = torch.ones((yf.shape[0], T_pose), device=device)
                    wchg = None
                loss_pose  = huber_pose_loss(yf[:, :T_pose, :6], y[:, :T_pose, :6], sw_pose, wchg)
                if T_pose > 1:
                    loss_posem = huber_pose_loss(
                        (yf[:, 1:T_pose, :6] - yf[:, :T_pose-1, :6]),
                        (y[:,  1:T_pose, :6] - y[:,  :T_pose-1, :6]),
                        sw_pose[:, 1:], wchg
                    )

            # —— jerk（三阶）惩罚：仅作用 pose 前6维 ——
            loss_jerk = torch.tensor(0., device=device)
            if opt.jerk_weight > 0 and yf.shape[2] >= 6:
                if sw is not None:
                    loss_jerk = opt.jerk_weight * jerk_loss_pose(yf[:, :, :6], w=sw)
                else:
                    ones_w = torch.ones((yf.shape[0], yf.shape[1]), device=device)
                    loss_jerk = opt.jerk_weight * jerk_loss_pose(yf[:, :, :6], w=ones_w)

            # —— 自适应加速度惩罚（pose 前6维）——
            acc_adapt = torch.tensor(0., device=device)
            if yf.shape[2] >= 6 and yf.shape[1] >= 3:
                # 速度与加速度
                vel = yf[:, 1:, :6] - yf[:, :-1, :6]           # [B, T-1, 6]
                acc = vel[:, 1:, :] - vel[:, :-1, :]           # [B, T-2, 6]

                # 速度的 EMA：区分稳定/突变
                alpha = 0.9
                ema = torch.zeros_like(vel)
                ema[:, 0, :] = vel[:, 0, :]
                for t in range(1, vel.shape[1]):
                    ema[:, t, :] = alpha * ema[:, t-1, :] + (1 - alpha) * vel[:, t, :]

                # 偏离 -> 权重
                dev  = torch.abs(vel - ema)                    # [B, T-1, 6]
                w_acc = 0.7 + 0.6 * torch.sigmoid(5.0 * (dev - dev.mean(dim=(0,1), keepdim=True)))
                w_acc = torch.clamp(w_acc, 0.6, 1.4)           # 防极端
                w_acc = w_acc[:, 1:, :]                        # 与 acc 对齐，[B, T-2, 6]

                # 与 DCAW 时序权重对齐并求损失
                if opt.use_dcaw:
                    min_tw = min(acc.shape[1], s.shape[1]-2, m.shape[1]-2)
                    sw_acc = (s[:, 2:2+min_tw] * m[:, 2:2+min_tw]).unsqueeze(-1)  # [B, min_tw, 1]
                    acc_adapt = 0.08 * torch.mean(torch.abs(acc[:, :min_tw, :] * w_acc[:, :min_tw, :] * sw_acc))
                else:
                    acc_adapt = 0.08 * torch.mean(torch.abs(acc * w_acc))

            # —— 合成总损失 ——
            lossG = (
                loss_s + lossg_e + lossg_em +
                loss_au + loss_aum + loss_au_stft_val +
                loss_pose + loss_posem + loss_jerk +
                0.1*lossg_gan + lossg_gan_au45 +
                loss_tv + loss_feat_tv + loss_expr_energy + loss_stft +
                vel_cons + acc_adapt
            )

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

            if step % 60 == 0:
                print(f'epoch: {epoch} step: {step}')
                print(f'  loss_s: {loss_s.item():.4f}  lossg_e: {lossg_e.item():.4f}  lossg_em: {lossg_em.item():.4f}')
                print(f'  loss_au: {loss_au.item():.4f}  loss_aum: {loss_aum.item():.4f}  loss_au_stft: {loss_au_stft_val.item():.4f}')
                print(f'  lossg_gan: {lossg_gan.item():.4f}  lossg_gan_au45: {lossg_gan_au45.item():.4f}')
                print(f'  loss_pose: {loss_pose.item():.4f}  loss_posem: {loss_posem.item():.4f}  loss_jerk: {loss_jerk.item():.4f}')
                print(f'  loss_tv: {loss_tv.item():.6f}  loss_feat_tv: {loss_feat_tv.item():.6f}')
                print(f'  loss_expr_energy: {loss_expr_energy.item():.6f}  loss_stft: {loss_stft.item():.6f}')
                print(f'  vel_cons: {vel_cons.item():.6f}  acc_adapt: {acc_adapt.item():.6f}')

if __name__ == '__main__':
    main()
