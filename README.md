# Mitigating Dynamic Graph Distribution Shifts via Mixture of Variational Experts

This repository contains the official implementation of the paper "Mitigating Dynamic Graph Distribution Shifts via Mixture of Variational Experts".

## Requirements

**Main dependencies:**
- CUDA == 10.1
- Python == 3.8.12
- PyTorch == 1.9.1
- PyTorch-Geometric == 2.0.1

To install all required packages, run the following command in the project root directory:

```bash
pip install -r requirements.txt
```


## Datasets

MoVE supports dynamic link prediction on both discrete-time and continuous-time dynamic graphs.  
All datasets can be downloaded here:  
https://1drv.ms/u/c/1b2f69874f634cd8/IQBXefaMKTWIRpRCbaCs9byhAcH5yc1bYAgkN-7RyrfPZTE?e=RbBokn

Once downloaded, please organize the datasets into the following directories based on their type:

### Discrete-time dynamic graph datasets
(collab, yelp, act, collab_04, collab_08)

Place these in: MoVE_discrete_graph/data/

### Continuous-time dynamic graph datasets
(Review, Reddit, UNtrade, UNvote)

Place these in: MoVE_continuous_graph/processed_data/


---

## Training

### Continuous-Time Dynamic Graphs

Example of training MoVE on the Reddit dataset:

```bash
cd MoVE_continuous_graph
python train_link_prediction.py --dataset_name reddit --model_name MoVE --patch_size 2 --num_runs 5 --gpu 0
```

### Discrete-Time Dynamic Graphs

Example of running MoVE on the Collab dataset:

```bash
cd MoVE_discrete_graph/scripts
python main.py --mode=eval --use_cfg=1
```

## Acknowledgement

We sincerely appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/haonan-yuan/EAGLE

https://github.com/yule-BUAA/DyGLib

https://github.com/wondergo2017/DIDA
