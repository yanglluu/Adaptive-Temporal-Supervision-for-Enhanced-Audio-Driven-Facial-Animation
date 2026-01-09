# -- B 版：真实 FACIAL 工程版（motion-driven DCAW）

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dcaw_controller import DCAWController

def preprocess(audio_pkl, npz, csv):
    audio = pickle.load(open(audio_pkl,"rb"),encoding="iso-8859-1")
    audio = np.asarray(audio)

    params = np.load(npz)["face"]
    blink = pd.read_csv(csv)["AU45_r"].values

    std = np.std(params, axis=0)
    mean = np.mean(params, axis=0)
    for i in range(6):
        params[:,i]=(params[:,i]-mean[i])/std[i]

    T=min(audio.shape[0],params.shape[0],blink.shape[0])
    audio=audio[:T]; params=params[:T]; blink=blink[:T].reshape(-1,1)

    params=np.concatenate([params[:, :6], blink, params[:, 7:71]], axis=1)

    return torch.tensor(audio), torch.tensor(params)

def extract_W(audio, params, win=128):
    T=len(audio)
    ctrl=DCAWController()
    W=[]
    for s in range(T-win):
        a=audio[s:s+win]
        p=params[s:s+win]
        W_t,_=ctrl.compute(a,p)
        W.append(W_t[win//2])
    return np.array(W)

def plot(W, out):
    plt.figure(figsize=(16,4))
    plt.step(np.arange(len(W)), W, where="mid", linewidth=2, color="r")
    plt.title("True DCAW Window Size Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Window Size (frames)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out,dpi=300)
    plt.show()
    print("[OK] Saved:", out)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audio_pkl", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="B_window.png")
    args=parser.parse_args()

    audio,params=preprocess(args.audio_pkl,args.npz,args.csv)
    W=extract_W(audio,params)
    plot(W,args.out)

if __name__=="__main__":
    main()
