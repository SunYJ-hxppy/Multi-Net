import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model_none
from util.visualizer import Visualizer
import torch,random,os
import numpy as np
from val import test, test_nf, seg_postprocessing, to_255
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision
from util.postprocessing import OnehotNcombine
import torch.nn as nn
from monai.transforms import Resize



def seed_torch(seed=1234):
    
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# def build_tensorboard(opt):
#     """Build a tensorboard logger."""
#     from logger import Logger
#     writer = Logger( opt )
#     return writer

if __name__ == '__main__':
    seed_torch()
    opt = TrainOptions().parse() 
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  
    name = opt.name
    print(name)
    print('The number of training dataset size = %d' % dataset_size)
    final = opt.n_epochs + opt.n_epochs_decay
    print(final)

    model = create_model_none(opt) 
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 0                
    tb_comment = f'Motion_Correction_synthesis_and Segmentation_240130'  
    writer = SummaryWriter( comment=tb_comment )

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): 
        print("epoch:", epoch)
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()  
        epoch_iter = 0                
        visualizer.reset()            

        # tensorboard 
        dis_total_running_loss = 0
        gen_total_running_loss = 0
        dis_total_epoch_loss   = 0
        gen_total_epoch_loss   = 0      
        seg_total_running_loss = 0
        seg_total_epoch_loss   = 0

        dict_dis_indiv_running_loss_tb = {}
        dict_gen_indiv_running_loss_tb = {}
        dict_seg_indiv_running_loss_tb = {}
        dict_dis_indiv_epoch_loss_tb = {}
        dict_gen_indiv_epoch_loss_tb = {}
        dict_seg_indiv_epoch_loss_tb = {}

        dict_merged_epoch_loss_tb = {}

        data_len = 1139
        
        with tqdm(total=data_len, desc=f'Epoch {epoch}/{final}', unit='images' ) as pbar:
            for i, data in enumerate(dataset):
                iter_start_time = time.time() 
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter  += opt.batch_size

                n_train = len(dataset)
                
                model.set_input(data)        
                model.optimize_parameters()  
                
                if total_iters % opt.display_freq == 0: 
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()

                if total_iters % opt.print_freq == 0:    
                    losses = model.get_current_losses()                                                                                                                                                                                                                                                                                                                                                                                                              
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                  
                if total_iters % opt.save_latest_freq == 0:  
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                # tensorboard 
                real_A, real_B, fake_B, pred_mask_clean_fake, dvf_flow, rec_B, warp_clean = model.return_img() 
                motion_structure, clean_structure, motion_artifact = model.return_progress_img()
                                
                motion_artifact0 = torch.Tensor(motion_artifact[0].cpu())
                pred_mask_clean_fake_copy = pred_mask_clean_fake.clone().detach()
                pred_mask_OnehotNcombine  = OnehotNcombine( nn.Softmax(dim=1)(pred_mask_clean_fake_copy) )[0].unsqueeze(0).detach().cpu()
                pred_mask_OnehotNcombine_4dim = pred_mask_OnehotNcombine.unsqueeze(0)
                pred_mask_OnehotNcombine3 = torch.cat((pred_mask_OnehotNcombine_4dim,pred_mask_OnehotNcombine_4dim,pred_mask_OnehotNcombine_4dim), dim=1)
                
                warp_clean_2 = torch.cat([warp_clean, warp_clean], dim=0)
                img_list = []
                progress_img_list = []
                dis_loss_tb, gen_loss_tb, seg_loss_tb = model.print_total_loss()

                "------------------TensorBoard image Plot-----------------"
                for imgs in [real_A, real_B, fake_B, rec_B, pred_mask_OnehotNcombine3, dvf_flow, warp_clean_2]: 
                    
                    if imgs.shape[0] >= 2:
                        imgs = imgs[0:1]

                    if imgs.shape[1] >= 2:
                        imgs = imgs[:,1:2]
                    
                    img_list.append(torch.Tensor(imgs[:3].cpu()))     
                        
                    output_img_tensor = torch.cat(img_list, dim=0)
                
                output_img_grid = torchvision.utils.make_grid( tensor    = output_img_tensor.data,
                                                               nrow      = 3,
                                                               padding   = 0,
                                                               normalize = True )
                
                writer.add_image( 'train/realA---realB---fakeB---rec_B---pred_mask_clean_fake---dvf_flow---warp_clean', output_img_grid, epoch )

                "------------------TensorBoard progress image Plot-----------------"
                for imgs1 in [real_A, motion_structure, clean_structure, motion_artifact0, fake_B, real_B]: 
                   
                    if imgs1.shape[0] >= 2:
                        imgs1 = imgs1[0:1]

                    if imgs1.shape[1] >= 2:
                        imgs1 = imgs1[:,1:2]
                    
                    resize_transform = Resize(spatial_size=real_A.shape[2:])
                    imgs1 = torch.stack([resize_transform(imgs1[i]) for i in range(imgs1.shape[0])])
                    
                    progress_img_list.append(torch.Tensor(imgs1[:3].cpu()))     
                    progress_img_tensor = torch.cat(progress_img_list, dim=0)
                
                progress_img_grid = torchvision.utils.make_grid( tensor    = progress_img_tensor.data,
                                                                 nrow      = 3, 
                                                                 padding   = 0,
                                                                 normalize = True )
                
                writer.add_image( 'train/progress_img---realA---motion_structure---clean_structure---motion_artifact---fakeB', progress_img_grid, epoch )
        
                "------------------TensorBoard Loss-----------------------"
                for key, val in dis_loss_tb.items(): 
                    if key not in dict_dis_indiv_running_loss_tb:
                        dict_dis_indiv_running_loss_tb[key] = val
                    else:
                        dict_dis_indiv_running_loss_tb[key] += val

                for key, val in dict_dis_indiv_running_loss_tb.items():
                    dict_dis_indiv_epoch_loss_tb[key] = val / n_train

                for key, val in gen_loss_tb.items(): 
                    if key not in dict_gen_indiv_running_loss_tb:
                        dict_gen_indiv_running_loss_tb[key] = val
                    else:
                        dict_gen_indiv_running_loss_tb[key] += val

                for key, val in dict_gen_indiv_running_loss_tb.items():
                    dict_gen_indiv_epoch_loss_tb[key] = val / n_train

                for key, val in seg_loss_tb.items(): 
                    if key not in dict_seg_indiv_running_loss_tb:
                        dict_seg_indiv_running_loss_tb[key] = val
                    else:
                        dict_seg_indiv_running_loss_tb[key] += val   

                for key, val in dict_seg_indiv_running_loss_tb.items():
                    dict_seg_indiv_epoch_loss_tb[key] = val / n_train

                dict_merged_epoch_loss_tb.update( dict_dis_indiv_epoch_loss_tb )
                dict_merged_epoch_loss_tb.update( dict_gen_indiv_epoch_loss_tb )
                dict_merged_epoch_loss_tb.update( dict_seg_indiv_epoch_loss_tb )

                for tag, value in dict_merged_epoch_loss_tb.items():
                    print(f"{tag}: {value:.4f}")
                    
                    writer.add_scalar(tag, value.item(), epoch+1) 
                    
                pbar.update(data['A'].shape[0]) 

                iter_data_time = time.time()
            model.update_learning_rate()   
            
            path_txt = os.path.join(r'./checkpoints', name, 'metric.txt')
            real_art = False
            if epoch > 20:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch) 
                if real_art:
                    test_nf(epoch, model, file_path = path_txt, phase='test')
                else:
                    test(epoch, model, file_path = path_txt, phase='test') 


            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
