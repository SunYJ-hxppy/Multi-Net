from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np


class ColorizationModel(Pix2PixModel):
  
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        Pix2PixModel.__init__(self, opt)
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']

    def lab2rgb(self, L, AB):
        
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def compute_visuals(self):
        self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)
        self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)
