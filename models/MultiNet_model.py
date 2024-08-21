import torch
import itertools, os
from util.util import normalize_image, normalize_image_pixel
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .ssim import *
from torch.cuda.amp import autocast, GradScaler
from .networks import init_net
from .MultiNet import Dual_Domain_GEN 
from .Joint_seg_moco.OptUNet_joint_model_scSE import OptUNet_motion
from monai.losses import DiceLoss
from util.loss.gratient_smoothing import GradientSmoothing, Grad
from util.loss.perceptual import PerceptualLoss
from util.loss.loss import MSELoss
from monai.losses.ssim_loss import SSIMLoss
from util.postprocessing import OnehotNcombine
import torch.nn as nn
from util.SpatialTransformer import SpatialTransformer
import numpy as np 

class MultiNetModel(BaseModel): 
    """
    This class implements the UDDN model, an unsupervised method to remove motion artifacts in MRI without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(no_dropout=True) 
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the UDDN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'idt_A', 'D_B', 'G_B', 'idt_B', 'seg_clean_fake', 'seg_clean_fake_ce', 'cycle_B', 'gradient_smooth', 'art', 'stn_clean']
        
        if self.isTrain:
            visual_names_A = ['real_A', 'fake_B', 'pred_mask_clean_fake', 'dvf_flow'] 
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'warp_clean'] 

        else:
            visual_names_A = ['real_A', 'fake_B', 'pred_mask_clean_fake', 'dvf_flow']
            visual_names_B = ['real_B', 'warp_clean']

        if self.isTrain and self.opt.lambda_identity > 0.0:  
            visual_names_A.append('idt_B') 
            visual_names_B.append('idt_A') 

        self.visual_names = visual_names_A + visual_names_B  
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B', 'joint_moco_seg_model']
            
        else: 
            self.model_names = ['G', 'joint_moco_seg_model']

        self.amp = False          

        self.netG = Dual_Domain_GEN()
        self.netjoint_moco_seg_model  = OptUNet_motion( spatial_dims = 2,
                                                        in_channels  = 2,
                                                        out_channels = 4,
                                                        deep_supervision=False ).to(self.device) 

        self.spatial_transform = SpatialTransformer(size=(256,256))
        
        init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids) 

        if self.isTrain: 
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  
            self.fake_B_pool = ImagePool(opt.pool_size) 

            self.criterionGAN    = networks.GANLoss(opt.gan_mode).to(self.device) 
            self.criterionCycle  = torch.nn.L1Loss()
            self.criterionIdt    = torch.nn.L1Loss()
            self.criterionart    = torch.nn.L1Loss()
            self.criterionwarp   = torch.nn.L1Loss()
            self.criterionwarp2  = torch.nn.MSELoss()
            self.criterionCycle1 = torch.nn.L1Loss()
            self.criterionCycle2 = torch.nn.MSELoss()
            self.diceloss         = DiceLoss(include_background=True, to_onehot_y=False, softmax=True, smooth_nr=1e-05, smooth_dr=1e-05)
            self.crossEntropyloss = nn.CrossEntropyLoss()
            self.smooth_loss    = Grad('l2', loss_mult=2).loss 
            self.Perceptualloss = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50").to(self.device)
            self.SSIMloss       = SSIMLoss(spatial_dims=2, data_range=1)
           
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_joint_moco_seg  = torch.optim.Adam(self.netjoint_moco_seg_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.amp:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            else:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_A)

                self.optimizers.append(self.optimizer_D_B)
            if self.amp:
                self.amp_scaler = GradScaler(enabled=True)

        # Tensorboard 
        self.dis_loss_tb = {}
        self.gen_loss_tb = {}
        self.seg_loss_tb = {}

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A      = input['A' if AtoB else 'B'].to(self.device)
        self.real_B      = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask_gt     = input['mask_gt'].to(self.device)
        self.mask_motion = input['mask_motion'].to(self.device)
        self.original_img_size = input['original_img_size']

    def forward(self):
        self.fake_B, self.fake_A, self.motion_structure, self.clean_structure, self.motion_artifact = self.netG.forward1(self.real_A, self.real_B)      
        self.rec_B, _, _, _, _     = self.netG.forward1(self.fake_A, self.fake_B) 
        real_A_unsqueeze          = self.real_A.unsqueeze(1)
        self.fake_B_copy          = self.fake_B.clone().detach() 
        fake_B_copy_norm          = normalize_image_pixel(self.fake_B_copy)
        fake_B_unsqueeze          = fake_B_copy_norm.unsqueeze(1)
        self.joint_input          = torch.cat([fake_B_unsqueeze, real_A_unsqueeze], dim=1) 
        self.pred_mask_clean_fake = self.netjoint_moco_seg_model.forward(self.joint_input)['pred_seg']
        self.dvf_flow             = self.netjoint_moco_seg_model.forward(self.joint_input)['pred_dvf']
        self.dvf_flow_copy        = self.dvf_flow.clone().detach()
        self.dvf_flow_copy1       = self.dvf_flow_copy[:,1,:,:].unsqueeze(0) 
        self.dvf_flow_copy_gpu    = self.dvf_flow_copy.to(self.device)
        self.warp_clean           = self.spatial_transform( self.real_A, self.dvf_flow_copy_gpu )
        
    def return_img(self): 
       
        return self.real_A, self.real_B, self.fake_B, self.pred_mask_clean_fake, self.dvf_flow, self.rec_B, self.warp_clean
   
    def return_progress_img(self):

        return self.motion_structure, self.clean_structure, self.motion_artifact

    def backward_D_basic(self, netD, real, fake):
        pred_real   = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake   = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.amp:
            self.amp_scaler.scale(loss_D).backward()
        else:
            loss_D.backward()

        return loss_D

    def backward_D_A(self): 
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A   = self.opt.lambda_A
        lambda_B   = self.opt.lambda_B
        
        if lambda_idt > 0:
            self.idt_B, self.idt_A = self.netG.forward2(self.real_A, self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.artifact = torch.abs(self.real_A - self.real_B)
        self.artifact_norm = normalize_image(self.artifact)
        self.dvf_copy_mean = torch.mean(self.dvf_flow_copy, dim=1)
        self.dvf_copy_norm = normalize_image(self.dvf_copy_mean)
        
        self.loss_art = self.criterionart(self.dvf_copy_norm, self.artifact_norm[:,1,:,:]) 
     
        if self.amp:
            self.real_A = self.real_A.half()
            self.real_B = self.real_B.half()

        self.loss_ssim = (1 - ms_ssim(self.real_B, self.rec_B)) * 10 
                
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_B + self.loss_ssim + self.loss_art 
        if self.amp:
            self.amp_scaler.scale(self.loss_G).backward()
        else:
            self.loss_G.backward()
        
    def backward_joint_moco_seg(self):
        gt_mask_center           = self.mask_gt[:,:,1,:,:]
        gt_mask_center_long      = gt_mask_center.long()
        gt_mask_center_long_3dim = torch.argmax(gt_mask_center_long, dim=1)

        self.loss_seg_clean_fake    = self.diceloss(self.pred_mask_clean_fake, gt_mask_center)
        self.loss_seg_clean_fake_ce = self.crossEntropyloss(self.pred_mask_clean_fake, gt_mask_center_long_3dim)
        total_seg_loss              = 3*self.loss_seg_clean_fake + 3*self.loss_seg_clean_fake_ce

        self.loss_gradient_smooth   = self.smooth_loss(None, self.dvf_flow)
        self.loss_stn_clean         = self.criterionwarp2(self.real_B, self.warp_clean)
        self.loss_stn_clean_l1      = self.criterionwarp(self.real_B, self.warp_clean)
        total_reg_loss              = self.loss_gradient_smooth + 2*self.loss_stn_clean + 2*self.loss_stn_clean_l1

        self.loss_joint_moco_seg  = total_seg_loss + total_reg_loss
        if self.amp:
            self.amp_scaler.scale(self.loss_joint_moco_seg).backward()
        else:
            self.loss_joint_moco_seg.backward()
        
    def optimize_parameters(self):
        
        self.forward()     
        self.set_requires_grad([self.netD_A, self.netD_B], False)  
        self.optimizer_G.zero_grad() 
        self.optimizer_joint_moco_seg.zero_grad()

        if self.amp:
            with autocast(enabled=True):

                self.forward()
                self.backward_G()
                self.backward_joint_moco_seg()
                
            self.amp_scaler.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer_G)

            self.amp_scaler.unscale_(self.optimizer_joint_moco_seg)
            torch.nn.utils.clip_grad_norm_(self.netjoint_moco_seg_model.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer_joint_moco_seg)

            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            with autocast(enabled=True):
                self.backward_D_A()  
                self.backward_D_B()  

            self.amp_scaler.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer_D)

            self.amp_scaler.update()
        else:
            self.backward_G()  
            self.optimizer_G.step()  
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D_A.zero_grad()  
            self.optimizer_D_B.zero_grad()  
            self.backward_D_A()             
            self.backward_D_B()             
            self.optimizer_D_A.step()       
            self.optimizer_D_B.step()       
            self.backward_joint_moco_seg()
            self.optimizer_joint_moco_seg.step()
            
    def print_total_loss(self): 
        # Tensorboad visualize 
        self.dis_loss_tb['D_loss/trans_a'] = self.loss_D_A
        self.dis_loss_tb['D_loss/trans_b'] = self.loss_D_B
        
        "1. Translational loss"
        self.gen_loss_tb['G_loss/GAN_trans_A'] = self.loss_G_A
        self.gen_loss_tb['G_loss/GAN_trans_B'] = self.loss_G_B
        "2. Identical loss"
        self.gen_loss_tb['G_loss/Idt_A'] = self.loss_idt_A
        self.gen_loss_tb['G_loss/Idt_B'] = self.loss_idt_B
        "3. Segmentation loss"
        self.seg_loss_tb['Seg_loss/seg_fake_clean'] = self.loss_seg_clean_fake
        self.seg_loss_tb['Seg_loss/reg_motion_clean'] = self.loss_gradient_smooth

        return self.dis_loss_tb, self.gen_loss_tb, self.seg_loss_tb
        
