# Multi-Net

![image](https://github.com/looooooooooloo/Multi-Net/assets/105937714/9d03ff9f-59af-41f1-9cdd-fee7e4002484)

This project is developed to correct the motion artifacts and segment the brain tissue with motion artifacts. 

# How to use
- Train a model:
```
python train.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
- Test a model:
```
python test.py --dataset_mode SlicesConcat --name yourown_name --model MultiNet
```
