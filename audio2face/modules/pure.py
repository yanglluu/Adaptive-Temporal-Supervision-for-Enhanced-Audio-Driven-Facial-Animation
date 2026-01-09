# -*- coding: utf-8 -*-
"""
只输出自适应窗口（红线），不画语速。

输入:
    audio.pkl
    params.npz
    blink.csv

输出:
    window_only.png
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dcaw_controller import DCAWController


# ------------------ 数据预处理（与 dataset102 完全一致） ------------------
def preprocess(audio_pkl, npz_path, csv_path):
    audio = pickle.load(open(audio_pkl, "rb"), encoding="iso-8859-1")
    audio = np.asarray(audio)

    params = np.load(npz_path)["face"]
    blinkinfo = pd.read_csv(csv_path)
    au45 = blinkinfo["AU45_r"].values

    # pose 前6维标准化
    std1 = np.std(params, axis=0)
    mean1 = np.mean(params, axis=0)
    for i in range(6):
        params[:, i] = (params[:, i] - mean1[i]) / std1[i]

    T = min(audio.shape[0], params.shape[0], au45.shape[0])
    audio = audio[:T]
    params = params[:T]
    au45 = au45[:T].reshape(-1, 1)

    params_full = np.concatenate(
        [params[:, :6], au45, params[:, 7:71]], axis=1
    )

    return torch.tensor(audio, dtype=torch.float32), \
           torch.tensor(params_full, dtype=torch.float32)


# ------------------ 调用真实 DCAWController ------------------
def extract_W_curve(audio, params, win=128):
    ctrl = DCAWController()
    T = len(audio)

    W_list = []
    for start in range(0, T - win):
        a = audio[start:start + win]
        p = params[start:start + win]

        W_t, _ = ctrl.compute(a, p)
        center = win // 2
        W_list.append(W_t[center])

    return np.array(W_list)


# ------------------ 画纯红线图 ------------------
def plot_red(W, out):
    x = np.arange(len(W))

    plt.figure(figsize=(16, 4))
    plt.step(x, W, where="mid", color="red", linewidth=2)

    plt.title("Adaptive Window Size (True DCAW)", fontsize=18)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Window Size (frames)", fontsize=14)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"[OK] Saved → {out}")


# ------------------ 主函数 ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_pkl", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="window_only.png")
    args = parser.parse_args()

    audio, params = preprocess(args.audio_pkl, args.npz, args.csv)
    W = extract_W_curve(audio, params)
    plot_red(W, args.out)


if __name__ == "__main__":
    main()
