# [MICCAI' 24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning
🚀 This is an Pytorch inplementation of [MICCAI'24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning

This project is developed to correct the motion artifacts and segment the brain tissue with motion artifacts. 

![image](https://github.com/user-attachments/assets/9a2b2dc2-2ede-4f93-a90f-40cf39f64102)

# How to use
- Train a model:
```
python train.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
- Test a model:
```
python test.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
