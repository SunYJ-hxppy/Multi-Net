# [MICCAI' 24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning
🚀 This is an Pytorch inplementation of [MICCAI'24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning

This project is developed to correct the motion artifacts and segment the brain tissue with motion artifacts. 

![Fig2_ver2](https://github.com/user-attachments/assets/20f43344-6f37-46b9-a54f-e8351e84dea8)


# How to use
- Train a model:
```
python train.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
- Test a model:
```
python test.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
