# -*- coding: utf-8 -*-
"""
可视化真实 DCAW 窗口大小，并对不同窗口档位加浅色阴影美化。

输入：
  --audio_pkl  DeepSpeech 预处理后的 audio.pkl
  --npz        face 参数 npz（含 'face'）
  --csv        OpenFace 导出的 blink csv（含 AU45_r）
输出：
  一张带阴影的窗口曲线图 window_shaded.png
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dcaw_controller import DCAWController


# ---------- 1. 预处理：完全对齐 dataset102.py ----------
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


# ---------- 2. 调用真实 DCAWController，得到 W 曲线 ----------
def extract_W_curve(audio_t, params_t, win=128):
    ctrl = DCAWController()
    T = audio_t.shape[0]

    W_list = []
    for start in range(0, T - win):
        a_win = audio_t[start:start + win]
        p_win = params_t[start:start + win]

        W_t, _ = ctrl.compute(a_win, p_win)  # 真实调用
        center = win // 2
        W_list.append(float(W_t[center]))

    return np.array(W_list, dtype=np.float32)


# ---------- 3. 画图：红线 + 阴影 ----------
def plot_window_with_shading(W, out_path="window_shaded.png"):
    T = len(W)
    x = np.arange(T)

    # 所有窗口候选
    uniq = sorted(list(set(W)))
    print("Detected window types:", uniq)

    # 为常见窗口值定义浅色
    palette = {
        64:  (0.80, 0.90, 1.00, 0.35),   # 浅蓝
        96:  (0.85, 1.00, 0.90, 0.35),   # 浅绿
        128: (1.00, 0.95, 0.80, 0.35),   # 浅橙
        192: (1.00, 0.85, 0.85, 0.35),   # 淡粉
        256: (0.95, 0.85, 1.00, 0.35),   # 淡紫
    }
    default_color = (0.88, 0.88, 0.88, 0.3)

    plt.figure(figsize=(18, 4))

    # 根据窗口值分段着色
    current = W[0]
    start = 0
    for i in range(1, T):
        if W[i] != current:
            color = palette.get(current, default_color)
            plt.axvspan(start, i, color=color, alpha=color[3])
            start = i
            current = W[i]
    # 最后一段
    color = palette.get(current, default_color)
    plt.axvspan(start, T, color=color, alpha=color[3])

    # 画红色窗口曲线
    plt.step(x, W, where="mid", color="red", linewidth=2, label="Window Size")

    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Window Size (frames)", fontsize=14)
    plt.title("Adaptive Window Size (True DCAW, with Shaded Segments)",
              fontsize=18)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[OK] Saved shaded window figure → {out_path}")


# ---------- 4. 主函数 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_pkl", required=True,
                        help="DeepSpeech audio feature .pkl")
    parser.add_argument("--npz", required=True,
                        help="Face param .npz (with key 'face')")
    parser.add_argument("--csv", required=True,
                        help="OpenFace blink csv (with column 'AU45_r')")
    parser.add_argument("--out", default="window_shaded.png")
    args = parser.parse_args()

    audio_t, params_t = preprocess(args.audio_pkl, args.npz, args.csv)
    W_curve = extract_W_curve(audio_t, params_t, win=128)
    plot_window_with_shading(W_curve, args.out)


if __name__ == "__main__":
    main()
