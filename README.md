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
  - [Code Details](#code-details)
  - [Robustness Evaluation](#robustness-evaluation)
- [Update Log](#update-log)
- [To-do](#to-do)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

![overview_v11](https://github.com/user-attachments/assets/784b54a8-7f25-4b9b-87b4-27808e10f1cb)

Adversarial training has achieved substantial performance in defending image retrieval against adversarial examples. However, existing studies in deep metric learning (DML) still suffer from two major limitations: weak adversary and model collapse. In this paper, we address these two limitations by proposing Collapse-Aware TRIplet DEcoupling (CA-TRIDE). Specifically, TRIDE yields a stronger adversary by spatially decoupling the perturbation targets into the anchor and the other candidates. Furthermore, CA prevents the consequential model collapse, based on a novel metric, collapseness, which is incorporated into the optimization of perturbation. We also identify two drawbacks of the existing robustness metric in image retrieval and propose a new metric for a more reasonable robustness evaluation. Extensive experiments on three datasets demonstrate that CA-TRIDE outperforms existing defense methods in both conventional and new metrics. 

## Features
- **Collapse-Aware**: Dynamically tracks collapseness to avoid model collapse.
- **Triplet Decouping**: Decouples attacks in Candidate Perturbation(CAP) and Anchor Perturbation(ANP) for stronger adversaries.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/michaeltian108/CA-TRIDE.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset Preparation
Download the datasets CUB, CARS and SOP from the official site or on Kaggle, and put them under the `datasets` folder, using the following names:
```
  cars/
  CUB_200_2011/
  Stanford_Online_Products/
```
### Training
To train the model through our CA-TRIDE, use `dataset:model:method' to train:
```bash
python train.py -C cub:rres18p:pgtripletN   #CUB
python train.py -C cars:rres18p:pgtripletN  #CARS
python train.py -C sop:rres18p:pgtripletN   #SOP
```
You can also change `rres18p` to other models to implement CA-TRIDE on other architecture, for example, use `rres50p` to train on ResNet50. (Note that we've only tested on ResNet18 models)

### Code Details
The main part of our code is located in `CA-TRIDE/defenses/pnp.py`. 

**Note:** use the corresponding `lambda` value before training. 

### Robustness Evaluation
To comprehensively evaluate the robustness of the trained model, you will need to run **10** different attacks upon the model through the following code:
```
#CA- Attack
python3 advrank.py -v -A CA:pm=-:W=1:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#CA+ Attack
python3 advrank.py -v -A CA:pm=+:W=1:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#QA- Attack
python3 advrank.py -v -A QA:pm=-:M=1:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#QA+ Attack
python3 advrank.py -v -A QA:pm=+:M=1:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#ES Attack
python3 advrank.py -v -A ES:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#TMA Attack
python3 advrank.py -v -A TMA:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#LTM Attack
python3 advrank.py -v -A LTM:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#GTM Attack
python3 advrank.py -v -A GTM:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]

#GTT Attack
python3 advrank.py -v -A GTT:eps=0.031372549:alpha=0.011764:pgditer=32 -C [YourModelCheckpoint]
```

After running each attack, you can calculate the **ERS** scores based on the paper [RobRank](https://github.com/cdluminate/robrank) or **ARS** according to [our paper](https://arxiv.org/abs/2312.07364).
(Note that the ES attack accounts for **2** attacks.)


## Update Log

### v1.0.0 🎉 08/01/2024
- Initial release of CA-TRIDE. 

### v1.1.0 ✨ 01/05/205
- Detailed tutorial on using CA-TRIDE provided.
- Enhanced Readme for readability and accessibility.

## To-do
- Code simplification.
- ...

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

## Acknowledgments
We would like to express our gratitude to  **@cdluminate** as our project is inspired and built upon the great work from at [RobRank](https://github.com/cdluminate/robrank). 
Thank you for your support and contributions!



