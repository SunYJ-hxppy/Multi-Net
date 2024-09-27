# [MICCAI' 24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning
🚀 This is an Pytorch inplementation of [MICCAI'24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning

This project is developed to correct the motion artifacts and segment the brain tissue with motion artifacts. 

![image](https://github.com/user-attachments/assets/aa3a7a85-af7c-4224-bf6b-45fb5e525d1a)

# How to use
- Train a model:
```
python train.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
- Test a model:
```
python test.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
