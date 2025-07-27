# Diffusion-Based Vision-Language Model for Zero-Shot Anomaly Detection in Medical Images

This is the Pytorch implementation of the paper:

**Diffusion-Based Vision-Language Model for Zero-Shot Anomaly Detection in Medical Images**

*Engineering Applications of Artificial Intelligence (EAAI), Under review.*

# Abstract
With the rapid advancement of diagnostic technology, the ability to detect pathological areas such as tumors and polyps has significantly improved. This progress provides medical imaging specialists with more precise visual information to support anomaly identification, diagnosis, treatment planning, and patient monitoring. However, existing unsupervised and semi-supervised anomaly detection methods struggle with data privacy constraints, limited annotated medical datasets, and challenges in generalization. Zero-Shot Anomaly Detection (ZSAD), which enables the detection of unseen categories without requiring class-specific training, has emerged as a promising solution by leveraging the vision-language alignment capabilities of Vision-Language Models (VLMs), such as Contrastive Language-Image Pretraining (CLIP). Despite recent progress, ZSAD remains hindered by high noise levels, sparse targets, and poor adaptability in complex medical imaging scenarios. To address these issues, we propose a novel framework: DiffusionCLIP, a diffusion-based VLM for zero-shot anomaly detection in two-dimensional medical images. Specifically, DiffusionCLIP integrates diffusion models into the VLM to progressively denoise multi-level features extracted from the CLIP visual encoder, enhancing feature robustness and discriminability. A multi-level feature fusion strategy is designed to aggregate multi-scale representations from different depths of the visual encoder, ensuring complementary semantic alignment across layers. In addition, a dynamically modulated weight loss function is introduced to adaptively balance the learning of hard and easy samples, further improving model generalization. Extensive experiments on multiple benchmark medical imaging datasets, demonstrate that the proposed method significantly outperforms existing zero-shot anomaly detection approaches in terms of accuracy, robustness, and generalization.

# Getting Started

## Installation
To set up the DiffusionCLIP environment, follow one of the methods below:

### Clone this repo:
```bash
git clone https://github.com/wossg-999/DiffusionCLIP.git && cd DiffusionCLIP
conda create -n DiffusionCLIP python=3.9.5 -y
conda activate DiffusionCLIP
pip install -r requirements.txt
```

# Preparation
dataset https://pan.baidu.com/s/1wEOFh-CIiLPbLfODk7E-QQ?pwd=ru4y code: ru4y 
# Training & Inference
```bash
CUDA_VISIBLE_DEVICES=$gpu_id python train.py
bash test.sh
```

# Acknowledgments
This project is heavily inspired by the excellent work from: [AdaCLIP by Yunkang Cao](https://github.com/caoyunkang/AdaCLIP). We sincerely thank the author for sharing their contributions.
# Citation
