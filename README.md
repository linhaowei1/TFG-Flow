# TFG-FLOW: Training-Free Guidance in Multi-Modal Generative Flow

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation for the ICLR 2025 paper **"TFG-FLOW: Training-Free Guidance in Multi-Modal Generative Flow"**

![Teaser Figure](assets/teaser.png) <!-- Replace with actual figure path -->

## 📌 Overview

This repository contains:

- ✅ Training-free guidance framework for multi-modal generative flows
- ✅ Implementation for quantum chemistry (QM9) property guidance
- ✅ Scripts for training:
  - Multi-modal flow models (`scripts/train_flow.sh`)
  - Guidance target predictors (`scripts/train_guide_clf.sh`)
  - Oracle target predictors (`scripts/train_oracle.sh`)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/linhaowei1/tfg-flow.git
cd tfg-flow
pip install -r requirements.txt

# Install PyTorch (example - modify for your cuda version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 🧠 Model Training

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

## 4 Inference with Guidance
```bash
for property in alpha cv gap homo lumo mu;
do
    bash scripts/qm9/guidance/qm9_$property.sh
done
```

## 📖 Citation
```bibtex
@inproceedings{tfgflow2025,
  title={TFG-Flow: Training-Free Guidance in Multi-Modal Generative Flow},
  author={Lin, Haowei and Li, Shanda and Ye, Haotian and Yang, Yiming and Ermon, Stefano and Liang, Yitao and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## 🤝 Contributing
Contributions welcome! Please open an issue or PR for suggestions/improvements.

## 📧 Contact
For questions, contact [YOUR_EMAIL] or open an issue.

## License
MIT License (see [LICENSE](LICENSE))
```

Key placeholders to replace:
1. Teaser figure path
2. Installation commands (PyTorch/CUDA versions)
3. Dataset download/preprocessing details
4. Property names in training scripts
5. Pretrained model links and metrics
6. Results table with actual numbers
7. Citation information
8. Contact information

Tips for completion:
- Add actual performance numbers from paper
- Include example generated molecules/structures if applicable
- Add architecture diagram in assets/
- Include link to paper PDF when available
- Add acknowledgments section if needed
- Include hardware requirements/benchmarks if relevant