# DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation
Official implementation of the paper submitted to **The Visual Computer**.

## 📌 Introduction
This repository contains the official code for our paper: DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation. Our method addresses the temporal non-uniformity of speech rhythm and sparse high-frequency facial events, achieving stable facial animation with eye blink consistency and robust to varying speaking rates.

## 🛠️ Environment Setup (环境配置，必写！保证可复现)
### Requirements
Python 3.8
PyTorch 1.13.1
OpenCV-python 4.7.0
numpy 1.24.3
pillow 9.5.0
# 具体配置请看requirements

### Installation
```bash
pip install -r requirements.txt

**Citation**

@article{Yang2026DCAWNet,
  title={DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation},
  author={Yang Mohan and Yang Haibo},
  journal={The Visual Computer},
  year={2026},
  note={Submitted}
}

**Acknowledgments**

We use Deep3DFaceReconstruction for face reconstruction, DeepSpeech and VOCA for audio feature extraction, and 3dface for face rendering. Rendering-to-video module borrows heavily FACIAL.
