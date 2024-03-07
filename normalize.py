import os
import numpy as np
import SimpleITK as sitk

def normalize_mm(img):     
    imin = np.percentile(img, 0.1)
    imax = np.percentile(img, 99.9)
    data = ((np.clip(img, imin, imax) - imin) * 255 / (imax - imin))
    return data
path = r'/public/huangmeiyan/wby/cycelegan/datasets/glioma_patch/trainA'
for root,dirs,files in os.walk(path):
    for files in files:
        path_file = os.path.join(root, files)
        if path_file.endswith('.nii.gz'):
            img = sitk.ReadImage(path_file)
            img1 = sitk.GetArrayFromImage(img)
            img1 = np.clip(img1,0,255.0)
            out1 = sitk.GetImageFromArray(img1)
            out1.SetOrigin(img.GetOrigin())
            out1.SetSpacing(img.GetSpacing())
            sitk.WriteImage(out1, path_file)

