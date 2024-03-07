
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, seg_postprocessing
from util import html
import cv2
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  
    
    opt.num_threads = 0  
    opt.batch_size = 1   
    opt.serial_batches = True  
    opt.no_flip = True   
    opt.eval = True
    opt.display_id = -1  
    dataset = create_dataset(opt) 

    model = create_model(opt)     
    model.setup(opt)              
    print('load model from the best model: ./checkpoints/my_brain/latest_net_G.pth')

    epoch_to_upload = opt.epoch_to_upload
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, epoch_to_upload))  
    if opt.load_iter > 0:  
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print()
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, epoch_to_upload))
    if opt.eval:
        model.eval()
    j = 0
    for i, data in enumerate(dataset):
       
        j = j+1

        model.set_input(data)  
        model.test()
        patient_id = data['patient'][0]
        print(patient_id)
       
        visuals = model.get_current_visuals() 
        
        img_path = model.get_image_paths()    
        if i % 5 == 0: 
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(webpage, visuals, img_path, data, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  
