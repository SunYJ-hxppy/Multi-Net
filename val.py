import os.path,random
import time,codecs
import numpy as np
import torch.utils.data
import collections
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset, my_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
from niqe import niqe
from pytorch_msssim import ms_ssim
import SimpleITK as sitk
from util.postprocessing import OnehotNcombine, Combine3D
import torch.nn.functional as F
from monai.transforms import Resize
from monai.losses import DiceLoss
from util.loss.gratient_smoothing import GradientSmoothing
from data.SlicesConcatTest_dataset import SlicesConcatTestdataset
from data import create_test_dataset


def to_255(img):
    return ((img + 1) / 2) * 255

def to_image( img, save_path, img_type, index):
    
    img = img * 255
    out = sitk.GetImageFromArray(img)
    out.SetOrigin(out.GetOrigin())
    out.SetSpacing(out.GetSpacing()) 
    image_path = os.path.join(save_path, '{}_'.format(img_type)+'{}.nii.gz'.format(index))
    sitk.WriteImage(out, image_path)

def to_numpy(tensor):
    img_numpy = tensor.detach().cpu().numpy()
    img_numpy = img_numpy.squeeze()
    img = img_numpy.squeeze()
    img = np.clip(img,0,255.0)
    img = img/255.0

    return img

def seg_postprocessing(pred_masks, data):
    
    probs = F.softmax( pred_masks, dim=1 )

    resize_probs_class0 = Resize( spatial_size=data['original_img_size'], mode='area')(probs[:,0,:,:].cpu().numpy())
    resize_probs_class1 = Resize( spatial_size=data['original_img_size'], mode='area')(probs[:,1,:,:].cpu().numpy())
    resize_probs_class2 = Resize( spatial_size=data['original_img_size'], mode='area')(probs[:,2,:,:].cpu().numpy())
    resize_probs_class3 = Resize( spatial_size=data['original_img_size'], mode='area')(probs[:,3,:,:].cpu().numpy())

    # resize_probs_class0 = torch.from_numpy(resize_probs_class0)
    # resize_probs_class1 = torch.from_numpy(resize_probs_class1)
    # resize_probs_class2 = torch.from_numpy(resize_probs_class2)
    # resize_probs_class3 = torch.from_numpy(resize_probs_class3)

    resize_probs = torch.stack((resize_probs_class0, resize_probs_class1, resize_probs_class2, resize_probs_class3), dim=1)
    probs_onehotNcombine = OnehotNcombine( resize_probs )      
    probs_onehotNcombine = np.array(probs_onehotNcombine.squeeze(0).cpu())

    return probs_onehotNcombine


def test(epochi, model, file_path, phase):

    model.eval()

    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.serial_batches = True
    opt.phase = phase
    name = opt.name
    dataset1 = create_test_dataset(opt)
    dataset1size = len(dataset1)
    final = opt.n_epochs + opt.n_epochs_decay
    path_val = os.path.join(r'./checkpoints', name, phase) 
    if not os.path.exists(path_val):
        os.makedirs(path_val)
        print('made checkpoints for val')

    print('The number of {} images = '.format(phase),  dataset1size)
    p1, p2   = 0, 0
    s1, s2   = 0, 0
    seg, reg = 0, 0
    for i, data in enumerate(dataset1):
        index = str(data['A_paths'])
        index = index.split('/')[-1]
        index = index.split('.')[0]

        patient_id = str(data['patient'])
        slice_num = str(data['slice_num'])
        mask_gt = data['mask_gt']

        model.set_input(data)
        model.test()
        realA, realB, fakeB, pred_mask_clean_fake, dvf_flow, rec_B, warp_clean = model.return_img()

        realA      = to_255(realA)
        realB      = to_255(realB)
        fakeB      = to_255(fakeB)
        recB       = to_255(rec_B)
        dvf_flow   = to_255(dvf_flow)
        warp_clean = to_255(warp_clean)
        
        ms1 = ms_ssim(realA, realB, data_range = 255, size_average = True)
        ms2 = ms_ssim(fakeB, realB, data_range = 255, size_average = True)

        s1 += ms1.item()
        s2 += ms2.item()

        realA = to_numpy(realA)
        realB = to_numpy(realB)
        fakeB = to_numpy(fakeB)
        recB  = to_numpy(recB)
        dvf_flow   = to_numpy(dvf_flow)
        warp_clean = to_numpy(warp_clean)
        
        print(realA.min(), realA.max())
        print(realB.min(), realB.max())
        print(fakeB.min(), fakeB.max())

        p1 += psnr(realA, realB)
        p2 += psnr(realB, fakeB)

        p1_all  = p1  / dataset1size
        p2_all  = p2  / dataset1size
        s1_all  = s1  / dataset1size
        s2_all  = s2  / dataset1size
        
        pred_mask_clean_fake  = seg_postprocessing(pred_mask_clean_fake, data)
        
        path_epoch = os.path.join(path_val,str(epochi), patient_id)
        
        if epochi == final:   
            if not os.path.exists(path_epoch):
                os.makedirs(path_epoch) 
            to_image(realA, path_epoch, img_type = patient_id + '_' + slice_num + '_real_A', index=index)
            to_image(realB, path_epoch, img_type = patient_id + '_' + slice_num + '_real_B', index=index)
            to_image(fakeB, path_epoch, img_type = patient_id + '_' + slice_num + '_fake_B', index=index)
            to_image(pred_mask_clean_fake, path_epoch, img_type  = patient_id + '_' + slice_num + '_pred_seg_clean_fake', index=index)
            to_image(dvf_flow, path_epoch, img_type = patient_id + '_' + slice_num + '_pred_dvf_flow', index=index)
            to_image(recB, path_epoch, img_type = patient_id + '_' + slice_num + '_rec_B', index=index)
            to_image(warp_clean, path_epoch, img_type = patient_id + '_' + slice_num + '_warp_clean', index=index)

 
    print('PSNR_ORI:{}'.format(p1_all), 'PSNR_AFT:{}'.format(p2_all), 'SSIM_ORI:{}'.format(s1_all),
          'SSIM_AFT:{}'.format(s2_all))

    with codecs.open(file_path, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(
            '\n' + '----------------------------------------------------------------------------')
        file_txt.write(
            '\n' + '{}_psnr_after:'.format(epochi) + str(p2_all) + '{}_ssim_after:'.format(epochi) + str(s2_all))


def test_nf(epochi,model,file_path,phase):
    model.eval()
    patient_id = data.split('_')[0]
    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.serial_batches = True
    opt.phase = phase
    name = opt.name
    dataset1 = create_dataset(opt)
    dataset1size = len(dataset1)
    final = opt.n_epochs + opt.n_epochs_decay
    path_val = os.path.join(r'./checkpoints', name, phase)
    if not os.path.exists(path_val):
        os.makedirs(path_val)
        print('i am here')

    print('The number of {} images = '.format(phase), dataset1size)
    niqe1, niqe2 = 0, 0
    s1, s2 = 0, 0
    for i, data in enumerate(dataset1):
        index = str(data['A_paths'])
        index = index.split('/')[-1]
        index = index.split('.')[0]

        model.set_input(data)
        model.test()
        realA, _, fakeB, pred_mask_clean_fake, dvf_flow = model.return_img()
       
        realA = to_numpy(realA)
        fakeB = to_numpy(fakeB)

        niqe1 = niqe1 + niqe(realA.astype('uint8'))
        niqe2 = niqe2 + niqe(fakeB.astype('uint8'))

        path_epoch = os.path.join(path_val, str(epochi), patient_id)
        
        if epochi == final: 
            if not os.path.exists(path_epoch):
                os.makedirs(path_epoch)
            to_image(realA, path_epoch, img_type = patient_id + '_real_A', index=index)
            to_image(fakeB, path_epoch, img_type = patient_id + '_fake_B', index=index)
            to_image(pred_mask_clean_fake, path_epoch, img_type = patient_id + '_pred_seg_clean_fake', index=index)
            to_image(dvf_flow, path_epoch, img_type = patient_id + '_pred_dvf_flow', index=index)

    niqe1_all = niqe1 / dataset1size
    niqe2_all = niqe2 / dataset1size
    print('NIQE_ORI:{}'.format(niqe1_all), 'NIQE_AFT:{}'.format(niqe2_all))

    with codecs.open(file_path, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(
            '\n' + '----------------------------------------------------------------------------')
        file_txt.write(
            '\n' + 'NIQE_AFT:{}'.format(niqe2_all))




