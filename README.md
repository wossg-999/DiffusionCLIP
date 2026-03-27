🚀 DiffusionCLIP
Diffusion-based Vision-Language Model for Zero-Shot Anomaly Detection in Medical Images
---
⚙️ Environment Setup
This project uses the same environment configuration as AdaCLIP.
```bash
git clone https://github.com/your-username/DiffusionCLIP.git
cd DiffusionCLIP

conda create -n diffusionclip python=3.8
conda activate diffusionclip

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
---
📂 Dataset
```
datasets/
├── dataset_name/
│   ├── train/
│   ├── test/
│   └── ground_truth/
```
---
🚀 Training
```bash
python train.py --dataset <dataset_name> --config configs/xxx.yaml
```
---
🔍 Evaluation
```bash
python test.py --dataset <dataset_name> --checkpoint <path>
```
---
📄 Citation
```bibtex
@article{CHEN2025112181,
title = {Diffusion-based vision-language model for zero-shot anomaly detection in medical images},
journal = {Engineering Applications of Artificial Intelligence},
volume = {161},
pages = {112181},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.112181},
url = {https://www.sciencedirect.com/science/article/pii/S095219762502189X},
author = {Yanhui Chen and Hongkang Tao and Zan Yang and Yunkang Cao and Chen Jiang and Longhua Hu and Pengwen Xiong and Haobo Qiu},
}
```
