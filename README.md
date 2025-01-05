# Collapse-Aware Triplet Decoupling for Adversarially Robust Image Retrieval (CA-TRIDE)

## Overview
This repository provides the official implementation of **"Collapse-Aware Triplet Decoupling for Adversarially Robust Image Retrieval (CA-TRIDE)"**. The paper introduces a novel approach to enhance adversarial robustness in image retrieval systems through the innovative concept of collapse-aware triplet decoupling. You can find the paper [here](https://arxiv.org/abs/2312.07364).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Update Log](#update-log)
- [Citation](#citation)
- [License](#license)

## Introduction
Adversarial attacks pose a significant challenge to image retrieval systems, leading to inaccurate or maliciously manipulated results. CA-TRIDE proposes a robust framework to mitigate these challenges by leveraging collapse-aware mechanisms and triplet loss decoupling. Our method achieves:

- Enhanced retrieval robustness under adversarial settings.
- Improved performance on benign examples while maintaining computational efficiency.

## Features
- **Collapse-Aware**: Dynamically tracks collapseness to avoid model collapse.
- **Triplet Decouping**: Decouples attacks in Candidate Perturbation(CAP) and Anchor Perturbation(ANP) for stronger adversaries.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ca-tride.git
   cd ca-tride
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset Preparation
Download the datasets CUB, CARS and SOP from the official site or on Kaggle.


### Training
To train the model:
```bash
python train.py --config config.yaml
```
Customize training parameters such as batch size, learning rate, and epochs in `config.yaml`.


## Results
The proposed CA-TRIDE achieves state-of-the-art adversarial robustness on standard benchmarks. Detailed results can be found in the `results/` folder and in our paper [here](https://arxiv.org/abs/2312.07364).

## Update Log

### v1.0.0 🎉 08/01/2024
- Initial release of CA-TRIDE. 

### v1.1.0 ✨ 01/05/205
- Detailed tutorial of using CA-TRIDE provided.
- Enhanced Readme for readability and accessibility.

##To-do
- Code simplification for
...

## Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{
tian2024collapseaware,
title={Collapse-Aware Triplet Decoupling for Adversarially Robust Image Retrieval},
author={Qiwei Tian and Chenhao Lin and Zhengyu Zhao and Qian Li and Chao Shen},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=cy3JBZKCw1}
}
```

## License
This project is licensed under the [MIT License](LICENSE).

