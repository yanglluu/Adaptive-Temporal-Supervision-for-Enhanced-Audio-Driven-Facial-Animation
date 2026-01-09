# -*- coding: utf-8 -*-
"""
从 FACIAL 的预处理结果生成美化后的
  - 语速曲线（上）
  - 自适应窗口曲线（下）

特点：
  * 上图：蓝色语速曲线，阴影只在曲线下方，
          阴影颜色与下图窗口大小 (256/128/64) 对齐
  * 下图：没有红线，用蓝/绿/粉深色线表示窗口，
          同时在线以下填充浅色阴影

用法示例：
python visualize_speed_window_beautified_fill.py \
  --audio_pkl /root/FACIAL-main/examples/audio_preprocessed/test1.pkl \
  --npz      /root/FACIAL-main/video_preprocess/train1.npz \
  --csv      /root/FACIAL-main/video_preprocess/openface/output.csv \
  --out      speed_window_beautified_fill.png
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dcaw_controller import DCAWController


# ---------------------------------------------------------
# 1. 预处理：完全对齐 dataset102.py
# ---------------------------------------------------------
def preprocess(audio_pkl, npz_path, csv_path):
    # audio: DeepSpeech 特征
    audio = pickle.load(open(audio_pkl, "rb"), encoding="iso-8859-1")
    audio = np.asarray(audio)

    # params: [T, 71]
    params = np.load(npz_path)["face"]

    # blink: AU45_r
    blink_df = pd.read_csv(csv_path)
    au45 = blink_df["AU45_r"].values

    # pose 前 6 维标准化
    std1 = np.std(params, axis=0)
    mean1 = np.mean(params, axis=0)
    for i in range(6):
        params[:, i] = (params[:, i] - mean1[i]) / (std1[i] + 1e-8)

    # 对齐长度
    T = min(audio.shape[0], params.shape[0], au45.shape[0])
    audio = audio[:T]
    params = params[:T]
    au45 = au45[:T].reshape(-1, 1)

    # 插入 AU45
    params_full = np.concatenate(
        [params[:, :6], au45, params[:, 7:71]],
        axis=1
    )

    audio_t = torch.tensor(audio, dtype=torch.float32)
    params_t = torch.tensor(params_full, dtype=torch.float32)
    return audio_t, params_t


# ---------------------------------------------------------
# 2. 调用真实 DCAWController，得到窗口曲线 W(t)
# ---------------------------------------------------------
def extract_W_curve(audio_t, params_t, win=128):
    ctrl = DCAWController()
    T = audio_t.shape[0]

    W_list = []
    # dense 滑窗，与训练保持一致
    for start in range(0, T - win):
        a_win = audio_t[start:start + win]
        p_win = params_t[start:start + win]

        W_t, _ = ctrl.compute(a_win, p_win)
        center = win // 2
        W_list.append(float(W_t[center]))

    return np.array(W_list, dtype=np.float32)


# ---------------------------------------------------------
# 3. 从 audio 估计语速 proxy（和你之前蓝线风格一致）
# ---------------------------------------------------------
def estimate_speed_from_audio(audio_t, smooth_win=7):
    """
    使用 DeepSpeech 特征帧间 L2 差分作为语速 proxy，再平滑+归一化到 [0,1]
    """
    a = audio_t.detach().cpu().numpy()
    if a.ndim == 3:
        a_flat = a.reshape(a.shape[0], -1)
    elif a.ndim == 2:
        a_flat = a
    else:
        a_flat = a.reshape(a.shape[0], -1)

    diff = np.linalg.norm(a_flat[1:] - a_flat[:-1], axis=1)
    diff = np.concatenate([[diff[0]], diff])

    if smooth_win > 1:
        pad = smooth_win // 2
        kernel = np.ones(smooth_win) / smooth_win
        diff = np.convolve(
            np.pad(diff, (pad, pad), mode="edge"),
            kernel,
            mode="same"
        )[pad:-pad]

    p95 = np.percentile(diff, 95)
    speed = diff / (p95 + 1e-8)
    speed = np.clip(speed, 0.0, 1.0)
    return speed.astype(np.float32)


# ---------------------------------------------------------
# 4. 美化绘图：阴影只在曲线以下；下图无红线
# ---------------------------------------------------------

# 颜色方案：浅色阴影 + 深色线（上下图统一）
PALETTE = {
    256: {"face": (0.80, 0.90, 1.00, 0.35), "edge": (0.10, 0.35, 0.80)},  # 蓝
    128: {"face": (0.85, 1.00, 0.90, 0.35), "edge": (0.10, 0.55, 0.10)},  # 绿
    64:  {"face": (1.00, 0.88, 0.92, 0.35), "edge": (0.70, 0.10, 0.30)},  # 粉
}
DEFAULT_FACE = (0.90, 0.90, 0.90, 0.25)
DEFAULT_EDGE = (0.40, 0.40, 0.40)


def plot_speed_window_beautified_fill(speed, W, out_path="beautified_fill.png"):
    T = min(len(speed), len(W))
    speed = np.array(speed[:T])
    W = np.array(W[:T])
    frames = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    # ---------- 上图：语速曲线 ----------
    ax1.plot(frames, speed, color="#1f77b4", linewidth=2.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Speaking Rate (normalized)", fontsize=13)
    ax1.set_title("Estimated Speaking Rate", fontsize=16)
    ax1.grid(alpha=0.3)

    # 阴影只在曲线下方，用窗口 W 的区间颜色
    start = 0
    current = W[0]
    for i in range(1, T + 1):
        if i == T or W[i] != current:
            colors = PALETTE.get(int(current), {"face": DEFAULT_FACE, "edge": DEFAULT_EDGE})
            face = colors["face"]
            ax1.fill_between(
                frames[start:i],
                0,
                speed[start:i],
                color=face,
                step="mid"
            )
            start = i
            if i < T:
                current = W[i]

    # 再把语速曲线画在阴影上面
    ax1.plot(frames, speed, color="#1f77b4", linewidth=2.0)

    # ---------- 下图：窗口曲线（无红线） ----------
    ax2.set_ylabel("Window Size (frames)", fontsize=13)
    ax2.set_title("Adaptive Window Size (True DCAW)", fontsize=16)
    ax2.set_xlabel("Frame", fontsize=13)
    ax2.grid(alpha=0.3)

    start = 0
    current = W[0]
    for i in range(1, T + 1):
        if i == T or W[i] != current:
            colors = PALETTE.get(int(current), {"face": DEFAULT_FACE, "edge": DEFAULT_EDGE})
            face = colors["face"]
            edge = colors["edge"]

            # 阴影：只在窗口线以下填充
            ax2.fill_between(
                frames[start:i],
                0,
                W[start:i],
                color=face,
                step="mid"
            )
            # 深色线：窗口边界
            ax2.step(
                frames[start:i],
                W[start:i],
                where="mid",
                color=edge,
                linewidth=2.0
            )

            start = i
            if i < T:
                current = W[i]

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"[OK] saved beautified figure to {out_path}")


# ---------------------------------------------------------
# 5. 主函数
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_pkl", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="speed_window_beautified_fill.png")
    args = parser.parse_args()

    audio_t, params_t = preprocess(args.audio_pkl, args.npz, args.csv)
    W_curve = extract_W_curve(audio_t, params_t, win=128)
    speed_full = estimate_speed_from_audio(audio_t)

    plot_speed_window_beautified_fill(speed_full, W_curve, args.out)


if __name__ == "__main__":
    main()
