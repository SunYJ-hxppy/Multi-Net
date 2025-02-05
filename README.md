# [MICCAI' 24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning
🚀 This is an Pytorch inplementation of [MICCAI'24] Deformation-Aware Segmentation Network Robust to Motion Artifacts for Brain Tissue Segmentation using Disentanglement Learning

This project is developed to correct the motion artifacts and segment the brain tissue with motion artifacts. 

# Method
![Fig2_ver2_git](https://github.com/user-attachments/assets/b64dcf3f-ead3-4f9d-90ec-1601da68931f)

# Result
![Fig6_ver5_git](https://github.com/user-attachments/assets/5deaf145-5831-4895-aa09-7fa40822aed7)
![Fig7_ver2_git](https://github.com/user-attachments/assets/4c35af7a-5048-4494-afb5-806bedaab72e)


# How to use
- Train a model:
```
python train.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
- Test a model:
```
python test.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
