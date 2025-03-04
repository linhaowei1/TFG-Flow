# TFG-FLOW: Training-Free Guidance in Multi-Modal Generative Flow

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

![Teaser Figure](assets/teaser.png) <!-- Replace with actual figure path -->

Official implementation for the ICLR 2025 paper **"TFG-Flow: Training-Free Guidance in Multi-Modal Generative Flow"** by Haowei Lin, Shanda Li, Haotian Ye, Yiming Yang, Stefano Ermon, Yitao Liang, Jianzhu Ma.

## 📖 Table of Contents
- [TFG-FLOW: Training-Free Guidance in Multi-Modal Generative Flow](#tfg-flow-training-free-guidance-in-multi-modal-generative-flow)
  - [📖 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [✨ Features](#-features)
  - [💻 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [📂 Dataset Preparation](#-dataset-preparation)
  - [🏋️ Model Training](#️-model-training)
    - [Training Pipeline](#training-pipeline)
    - [1. Train Base Flow Model](#1-train-base-flow-model)
    - [2. Train Guidance Classifier](#2-train-guidance-classifier)
    - [3. Train Oracle Model](#3-train-oracle-model)
  - [🔮 Inference with Guidance](#-inference-with-guidance)
    - [Batch Generation](#batch-generation)
  - [🗃️ Pre-trained Models](#️-pre-trained-models)
  - [🤝 Contributing](#-contributing)
  - [📖 Citation](#-citation)
  - [📧 Contact](#-contact)

## 🌟 Overview

**TFG-Flow** introduces a novel training-free guidance framework for multi-modal generative flows, enabling controlled generation across diverse modalities without retraining. Our method achieves state-of-the-art performance in molecular property-guided generation on the QM9 quantum chemistry dataset, supporting precise control over 6 key molecular properties:

- Alpha
- Cv (Heat capacity)
- Gap (HOMO-LUMO gap)
- Homo (Highest occupied molecular orbital)
- Lumo (Lowest unoccupied molecular orbital)
- Mu (Dipole moment)

## ✨ Features

- 🚀 **Training-Free Guidance**: Modify generation behavior without model retraining
- 🔬 **Multi-Modal Flow**: Implement multimodal flow on molecule generation
  
## 💻 Installation

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (recommended)
- NVIDIA drivers compatible with CUDA 12.1

### Setup
```bash
# Create and activate conda environment
conda create -n tfgflow python=3.10 -y
conda activate tfgflow

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone repository and install dependencies
git clone https://github.com/linhaowei1/tfg-flow.git
cd tfg-flow
pip install -r requirements.txt
```

## 📂 Dataset Preparation

The QM9 dataset will be automatically downloaded and processed on first run. 

## 🏋️ Model Training

### Training Pipeline
1. **Base Flow Model**: Learn the underlying data distribution
2. **Guidance Classifier**: Estimate conditional probabilities for guidance
3. **Oracle Model**: Provide target property predictions 

### 1. Train Base Flow Model
```bash
bash scripts/train_flow.sh  
```

### 2. Train Guidance Classifier
```bash
bash scripts/train_guide_clf.sh
```

### 3. Train Oracle Model
```bash
bash scripts/train_oracle.sh  
```

## 🔮 Inference with Guidance

Generate molecules with specific property targets:

```bash
bash scripts/qm9/guidance/qm9_homo.sh
```

### Batch Generation
```bash
for property in alpha cv gap homo lumo mu; do
    bash scripts/qm9/guidance/qm9_${property}.sh
done
```

**Note:** For energy unit conversion between Hartree and meV:
```python
1 Hartree = 27211.4 meV  # for homo, lumo, gap
```

## 🗃️ Pre-trained Models

Access our pre-trained checkpoints:

```bash
./storage/
├── flow_models.pth       # Base flow checkpoints
├── guide_clf_ckpt.zip # Guidance predictors
└── oracle_clf_ckpt.zip     # Target property predictors
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


## 📖 Citation

If you use TFG-Flow in your research, please cite:

```bibtex
@inproceedings{tfgflow2025,
  title={TFG-Flow: Training-Free Guidance in Multi-Modal Generative Flow},
  author={Lin, Haowei and Li, Shanda and Ye, Haotian and Yang, Yiming and Ermon, Stefano and Liang, Yitao and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## 📧 Contact

For questions and collaborations:
- Haowei Lin: [linhaowei@pku.edu.cn](mailto:linhaowei@pku.edu.cn)
- GitHub Issues: [https://github.com/linhaowei1/tfg-flow/issues](https://github.com/linhaowei1/tfg-flow/issues)

