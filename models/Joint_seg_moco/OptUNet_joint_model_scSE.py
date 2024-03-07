from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import pdb

from .OptUNet_blocks import UnetBasicBlock, UnetOutBlock, UnetUpBlock

import torch.nn as nn
import torch

class OptUNet_motion(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        deep_supervision: bool
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        
        self.cs_unit_encoder = []
        self.cs_unit_decoder = []
        
        self.input_conv_seg = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=3,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.seg_down1 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=64,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.seg_down2 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=96,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2
                                     )
        
        self.seg_bottleneck = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=128,
                                     out_channels=192,
                                     kernel_size=3,
                                     stride=2,
                                     )
        
        self.input_conv_moco = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=self.in_channels,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.moco_down1 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=64,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.moco_down2 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=96,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2
                                     )
        
        self.moco_bottleneck = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=128,
                                     out_channels=192,
                                     kernel_size=3,
                                     stride=2,
                                     )
        # "------------------------------------------------------------------------------------"         
        self.seg_up1 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=192,
                                out_channels=128,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.seg_up2 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=128,
                                out_channels=96,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )        
        self.seg_up3 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=96,
                                out_channels=64,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
               
        self.seg_out1 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=64,
                                  out_channels=self.out_channels,
                                  )
        self.seg_out2 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=96,
                                  out_channels=self.out_channels,
                                  )
        self.seg_out3 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=128,
                                  out_channels=self.out_channels,
                                  )
        # "------------------------------------------------------------------------------------"
        self.moco_up1 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=192,
                                out_channels=128,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )      
        self.moco_up2 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=128,
                                out_channels=96,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.moco_up3 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=96,
                                out_channels=64,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        
        self.cs_unit_encoder.append(nn.Parameter(0.5*torch.ones(96, 2, 2).cuda(), requires_grad=True))
        self.cs_unit_encoder.append(nn.Parameter(0.5*torch.ones(128, 2, 2).cuda(), requires_grad=True))
        self.cs_unit_encoder.append(nn.Parameter(0.5*torch.ones(192, 2, 2).cuda(), requires_grad=True))
        
        self.cs_unit_decoder.append(nn.Parameter(0.5 * torch.ones(128, 2, 2).cuda(), requires_grad=True))
        self.cs_unit_decoder.append(nn.Parameter(0.5 * torch.ones(96, 2, 2).cuda(), requires_grad=True))

        self.moco_out1 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=64,
                                  out_channels=2, 
                                  ) 
        self.moco_out2 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=96,
                                  out_channels=2,
                                  )
        self.moco_out3 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=128,
                                  out_channels=2,
                                  )
        
        self.cs_unit_encoder = torch.nn.ParameterList(self.cs_unit_encoder)
        self.cs_unit_decoder = torch.nn.ParameterList(self.cs_unit_decoder)

    def apply_cross_stitch(self, seg, moco, alpha):  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        shape = seg.shape
        newshape = [shape[0], shape[1],  shape[2] * shape[3]]  
              
        seg_flat = seg.view(newshape)  
        moco_flat = moco.view(newshape)

        seg_flat = torch.unsqueeze(seg_flat, -1)  
        moco_flat = torch.unsqueeze(moco_flat, -1)

        a_concat_b = torch.cat([seg_flat, moco_flat], dim=3)  
        alphas_tiled = alpha.unsqueeze(0) 
    
        alphas_tiled = alphas_tiled.to(device)
        a_concat_b = a_concat_b.to(device)
        a_concat_b_transpose = a_concat_b.transpose(3, 2)

        out = torch.matmul(alphas_tiled, a_concat_b_transpose)   
        out = out.permute(-2, 0, 1, -1)  
        out_seg  = out[0, :, :, :]  
        out_moco = out[1, :, :, :]  

        out_seg  = out_seg.view(shape)  
        out_moco = out_moco.view(shape) 

        return out_seg, out_moco
    
    def forward( self, input ):
        
        seg_x0 = self.input_conv_seg( input[:,0,:,:,:] ) 
        moco_x0 = self.input_conv_moco( input[:,:,1,:,:] ) 

        # Encoder
        # 1. Encoder: segmentation Encoder
        # 2. Encoder: Motion correction Encoder
        seg_x1  = self.seg_down1( seg_x0 ) 
        moco_x1 = self.moco_down1( moco_x0 )
        cs_seg1, cs_moco1 = self.apply_cross_stitch(seg_x1, moco_x1, self.cs_unit_encoder[0]) 

        seg_x2  = self.seg_down2( cs_seg1 ) 
        moco_x2 = self.moco_down2( cs_moco1 ) 
        cs_seg2, cs_moco2 = self.apply_cross_stitch(seg_x2, moco_x2, self.cs_unit_encoder[1])

        seg_x3 = self.seg_bottleneck( cs_seg2 ) 
        moco_x3 = self.seg_bottleneck( cs_moco2 )
        cs_seg3, cs_moco3 = self.apply_cross_stitch(seg_x3, moco_x3, self.cs_unit_encoder[2])

        # Decoder
        #1. Decoder: segmentation Decoder
        #2. Decoder: motion correction Decoder 
        seg_x4  = self.seg_up1( cs_seg3, cs_seg2 )    
        moco_x4 = self.moco_up1( cs_moco3, cs_moco2 ) 
        cs_seg4, cs_moco4 = self.apply_cross_stitch(seg_x4, moco_x4, self.cs_unit_decoder[0])

        seg_x5  = self.seg_up2( cs_seg4, cs_seg1 )  
        moco_x5 = self.moco_up2( cs_moco4, cs_moco1 )  
        cs_seg5, cs_moco5 = self.apply_cross_stitch(seg_x5, moco_x5, self.cs_unit_decoder[1])

        seg_x6  = self.seg_up3( cs_seg5, seg_x0 )  
        moco_x6 = self.moco_up3( cs_moco5, moco_x0 )  

        # Output
        output1 = self.seg_out1( seg_x6 )   
        output2 = self.moco_out1( moco_x6 ) 
      
        if self.training and self.deep_supervision:
            
            #1.decoder (seg)
            output3   = interpolate( self.seg_out2( seg_x5 ), output1.shape[2:]) 
            output4   = interpolate( self.seg_out3( seg_x4 ), output1.shape[2:])

            #2.decoder (correction)
            output5   = interpolate( self.moco_out2( moco_x5 ), output2.shape[2:])
            output6   = interpolate( self.moco_out3( moco_x4 ), output2.shape[2:])

            output_all = [ output1, output2, output3, output4, output5, output6 ]
            segmenation_output = [ output1, output3, output4 ]
            motion_output = [ output2, output5, output6 ]
            
            return { 'pred_seg' : torch.stack(segmenation_output, dim=1),
                     'pred_dvf' : torch.stack(motion_output, dim=1) }
               
        return { 'pred_seg': output1,
                 'pred_dvf' : output2}
        
    