# DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation
Official implementation of the paper submitted to ****

## 📌 Introduction
This repository contains the official code for our paper: DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation. Our method addresses the temporal non-uniformity of speech rhythm and sparse high-frequency facial events, achieving stable facial animation with eye blink consistency and robust to varying speaking rates.

## 📂 Repository Structure
DCAW-Net/
├── configs/ # Training/testing hyper-parameter config files

├── data/ # Data preprocessing codes and dataloader

├── models/ # Core DCAW-Net network architecture

├── utils/ # Loss functions, metrics, visualization tools

├── checkpoints/ # Pretrained model checkpoints

├── samples/ # Test audio samples and generated results

├── train.py # Training script

├── test.py # Inference and evaluation script

└── requirements.txt


## 🛠️ Environment Setup
### Requirements
Python 3.8
PyTorch 1.13.1
OpenCV-python 4.7.0
numpy 1.24.3
pillow 9.5.0
torchvision 0.14.1
librosa 0.10.1
scipy 1.10.1
matplotlib 3.7.1
tqdm 4.65.0

### Installation
```bash
pip install -r requirements.txt


📚 Dataset Preparation
Our experiments are conducted on the public benchmark datasets for audio-driven talking face generation (FACIAL https://github.com/zhangchenxu528/FACIAL,  VoxCeleb2). Download the official datasets and use the preprocessing scripts in data/ folder to extract audio features and facial geometric information for training and testing.

✏️ Citation
If you find this work helpful for your research, please cite our paper:
@article{Yang2026DCAWNet,
  title={DCAW-Net: Dual-Driven Adaptive Temporal Supervision for Audio-Driven Talking Face Generation},
  author={Yang Mohan and Yang Haibo},
  journal={},
  year={2026},
  note={Submitted}
}

🙏 Acknowledgments
We use Deep3DFaceReconstruction for face reconstruction, DeepSpeech and VOCA for audio feature extraction, and 3dface for face rendering. Rendering-to-video module borrows heavily FACIAL. We thank all the authors for their excellent open-source works.
📧 Contact
For any questions about the code, please contact the authors via email.
📄 License
This project is licensed under the MIT License.
