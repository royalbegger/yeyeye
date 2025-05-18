# GNN_LSAP


This repository contains the Python implementation of the framework described in: [Tackling the Linear Sum Assignment Problem with Graph Neural Networks](http://) - _Carlo Aironi, Samuele Cornell, and Stefano Squartini_, where Linear Sum Assignment Problems of different dimensions are faced with a data-driven approach based on Graph Neural Networks, and accuracy is compared against two existing DNN-based frameworks.

## Requirements
- scipy                     1.8.1
- pytorch                   1.11.0
- torch-geometric           2.0.4
- torch-cluster             1.6.0
- torch-scatter             2.0.9
- torch-sparse              0.6.13


# 生成训练数据
python main.py --mode generate --id exp1 --n 4 --s 1000 --fp ./data/train_4x4.npy

# 生成验证数据
python main.py --mode generate --id exp1 --n 4 --s 200 --fp ./data/val_4x4.npy

# 训练模型
python main.py --mode train --id exp1 --n 4 --h 32 --train_file ./data/train_4x4.npy --val_file ./data/val_4x4.npy

# 测试模型
python main.py --mode test --id exp1 --n 4 --test_file ./data/val_4x4.npy --test_chkpt ./logs/trained_net_exp1.pth
