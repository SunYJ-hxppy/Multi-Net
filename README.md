# Multi-Net

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
