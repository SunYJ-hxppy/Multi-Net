import nibabel as nib
from monai.transforms import LoadImage, Compose, NormalizeIntensityd, ScaleIntensityd, RandSpatialCropd, RandFlipd, Resized, \
                             RandRotate90d, Rand3DElasticd, RandAdjustContrastd, CenterSpatialCropd, RandRotated, \
                             ResizeWithPadOrCropd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import logging
import numpy as np
import os
import pdb
import re

import glob
import random
import os
import SimpleITK as sitk

from data.base_dataset import BaseDataset

import torch
import torchvision.transforms as transform
from data.image_folder import make_dataset

class SlicesConcatTestdataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_test = "/data/final_dataset_less_opendataset/test/"
        data_lists = os.listdir(self.dir_test)
        data_lists.sort()
        data_lists = [x for x in data_lists if x.endswith('s')==False]

        train_percent = 1.0
        self.train_data_lists = random.sample(data_lists[:], int(len(data_lists) * train_percent))
    
    @classmethod
    def preprocess(cls, data):
        
        transform = Compose([
            # NormalizeIntensityd( keys=['imgs'],
            #                     nonzero=False, channel_wise=True ),
            ScaleIntensityd( keys=['imgs'], minv=0.0, maxv=1.0, factor=None),
            Resized(keys=['imgs','masks'],
                    spatial_size=[3, 256, 256], # slice & H & W
                    mode=['area','nearest']), #mode=['area','area']
                ])    

        augmented_data = transform(data)        
        return augmented_data
            
    def __len__(self):
        return len( self.train_data_lists ) 
    
    def __getitem__(self, idx):
        slice_concat_data = self.train_data_lists[idx]
        patient_id = slice_concat_data.split("_")[0]
        slice_num = slice_concat_data.split('center')[1]
        slice_num = slice_num.split('.')[0]

        test_motion_data_path = "/data/slices3_concat_files_less_data/motion1_test/"
        test_clean_data_path  = "/data/slices3_concat_files_less_data/clean_test/"

        data_dir = os.path.join(self.dir_test, slice_concat_data)
        load_data = np.load(data_dir)
        _, _, original_h, original_w = torch.Tensor(load_data).shape

        clean_data   = load_data[:,0,:,:] # shape:(3,192,256)
        motion_data  = load_data[:,1,:,:]
        motion2_data = load_data[:,2,:,:]
        mask_data    = load_data[:,3,:,:]
        pred_motion_mask = load_data[:,3,:,:]

        clean_tensor   = torch.Tensor( clean_data ).unsqueeze(0)
        motion_tensor  = torch.Tensor( motion_data ).unsqueeze(0)
        motion2_tensor = torch.Tensor( motion2_data ).unsqueeze(0)
        mask_tensor    = torch.Tensor( mask_data ).unsqueeze(0)
        pred_motion_mask_tensor = torch.Tensor( pred_motion_mask ).unsqueeze(0)
 
        one  = torch.ones( mask_tensor.size() )
        zero = torch.zeros( mask_tensor.size() )
        
        class_BG  = torch.where( mask_tensor == 0, one, zero )
        class_csf = torch.where( mask_tensor == 1, one, zero )
        class_GM  = torch.where( mask_tensor == 2, one, zero ) # torch.where(condition, x, y) → Tensor (if condition → x, if not → y)
        class_WM  = torch.where( mask_tensor == 3, one, zero )

        class_BG_pred  = torch.where( pred_motion_mask_tensor == 0, one, zero )
        class_csf_pred = torch.where( pred_motion_mask_tensor == 1, one, zero )
        class_GM_pred  = torch.where( pred_motion_mask_tensor == 2, one, zero ) # torch.where(condition, x, y) → Tensor (if condition → x, if not → y)
        class_WM_pred  = torch.where( pred_motion_mask_tensor == 3, one, zero )

        train_masks_BG  = class_BG
        train_masks_csf = class_csf
        train_masks_GM  = class_GM
        train_masks_WM  = class_WM

        pred_mask_BG = class_BG_pred
        pred_mask_csf = class_csf_pred
        pred_mask_GM = class_GM_pred
        pred_mask_WM = class_WM_pred

        concat_tensor = torch.cat( (clean_tensor, motion_tensor, motion2_tensor,
            train_masks_BG, train_masks_csf, train_masks_GM, train_masks_WM,
            pred_mask_BG, pred_mask_csf, pred_mask_GM, pred_mask_WM ), 0 )
            
        data = {
            'imgs'  : np.array(concat_tensor[0:3,:,:,:]),
            'masks' : np.array(concat_tensor[3:,:,:,:])
        }
        
        augmented_imgs_masks = self.preprocess( data )
        augmented_imgs       = augmented_imgs_masks['imgs']
        augmented_masks      = augmented_imgs_masks['masks']

        normalized_imgs = (augmented_imgs - 0.5)/0.5
        
        ones_  = torch.ones( torch.Tensor(augmented_masks).size() )
        zeros_ = torch.zeros( torch.Tensor(augmented_masks).size() )
        processed_resized_concat_tensor = torch.where( torch.Tensor(augmented_masks) > 0.5, ones_, zeros_ )

        train_imgs  = np.array(normalized_imgs)    
        train_masks = np.array(processed_resized_concat_tensor)   

        clean_img_slices   = torch.from_numpy(train_imgs[0,:,:,:]).type(torch.FloatTensor)
        motion1_img_slices = torch.from_numpy(train_imgs[1,:,:,:]).type(torch.FloatTensor)
        motion2_img_slices = torch.from_numpy(train_imgs[2,:,:,:]).type(torch.FloatTensor)
        mask_slices_GT     = torch.from_numpy(train_masks[0:4,:,:,:]).type(torch.FloatTensor)
        mask_slices_pred   = torch.from_numpy(train_masks[4:,:,:,:]).type(torch.FloatTensor)

        return {
            'A': motion1_img_slices, 'B': clean_img_slices, 'mask_gt': mask_slices_GT, 'mask_motion': mask_slices_pred,
            'A_paths': test_motion_data_path, 'B_paths': test_clean_data_path, 
            'patient' : patient_id, 'original_img_size' : (original_h, original_w), 'slice_num' : slice_num
        }