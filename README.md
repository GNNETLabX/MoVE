# Mitigating Dynamic Graph Distribution Shifts via Mixture of Variational Experts

This repository contains the official implementation of the paper
“Mitigating Dynamic Graph Distribution Shifts via Mixture of Variational Experts”.

##  Requirements

Main dependencies:

- CUDA == 10.1
- Python == 3.8.12
- PyTorch == 1.9.1
- PyTorch-Geometric == 2.0.1

To install all required packages, run the following command in the project root directory:

```bash
pip install -r requirements.txt
```
## Training

MoVE supports dynamic link prediction on both discrete-time and continuous-time dynamic graphs.

**Example 1: Train MoVE on the Amazon_review dataset**

```bash
python train_continues.py --dataset_name AmazonReview --num_experts 4 --batch_size 200 --num_runs 5 --gpu 0
```
**Example 2: Train MoVE on the COLLAB dataset**
```bash
python train_discrete.py --dataset_name COLLAB --load_best_configs --num_experts 4 --num_runs 5
```
## Acknowledgement

We sincerely appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/haonan-yuan/EAGLE

https://github.com/yule-BUAA/DyGLib

https://github.com/wondergo2017/DIDA
