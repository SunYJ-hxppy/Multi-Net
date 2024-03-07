import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import functools
import  torch.fft

class Conv(nn.Module): 
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim  
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias) 
        self.relu = None 
        self.bn   = None 
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module): 
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.InstanceNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ChannelPool(nn.Module): 
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module): 
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        self.fc1 = nn.Conv2d(ch_2*2, ch_2*2 // r_2 , kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2*2 // r_2, ch_2*2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 *2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        
        g = torch.cat([g, x], 1)
        g_in = g

        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        x = g
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([ x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class LayerNorm(nn.Module): 
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
     zero = nn.ZeroPad2d,
  reflect = nn.ReflectionPad2d,
replicate = nn.ReplicationPad2d)

conv_dict = dict(
   conv2d = nn.Conv2d,
 deconv2d = nn.ConvTranspose2d)

norm_dict = dict(
     none = lambda x: lambda x: x,
 spectral = lambda x: lambda x: x,
    batch = nn.BatchNorm2d,
 instance = nn.InstanceNorm2d,
    layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
      relu = lambda: nn.ReLU(inplace=True),
      gelu = lambda: nn.GELU(),
     lrelu = lambda: nn.LeakyReLU(0.2,inplace=True),
     prelu = lambda: nn.PReLU(),
      selu = lambda: nn.SELU(),
      tanh = lambda: nn.Tanh())

class ConvolutionBlock(nn.Module): 
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)   
        self.conv = conv_dict[conv](**conv_opts)
        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels) 
        if norm == "spectral": self.conv = spectral_norm(self.conv)
        self.activ = activ_dict[activ]()

    def forward(self,x):
        return self.activ(self.norm(self.conv(self.pad(x))))

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block = block + [ConvolutionBlock(
                  in_channels = channels , out_channels=channels , kernel_size =3,
                  stride = 1, padding=1,norm=norm, activ=activ,pad=pad
        )]

        block = block + [ConvolutionBlock(
                  in_channels = channels , out_channels=channels , kernel_size =3,
                  stride = 1, padding=1,norm=norm, activ='none',pad=pad
        )]

        self.model = nn.Sequential(*block)

    def forward(self,x):
        return x + self.model(x)

class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm='none', activ='relu'):
        super(FullyConnectedBlock, self).__init__()

        self.fc =  nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral": self.fc = spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.fc(x)))


class Encoder1(nn.Module):
    def __init__(self,  base_ch, num_down, num_residual, res_norm='instance', down_norm='instance',input_ch= 3):
        super(Encoder1, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x):
        sides = []
        for layers_id, layer in enumerate(self.layers):
            x = layer(x)
            
        return x, sides[::-1] 
    
class Encoder2(nn.Module):
    
    def __init__(self,  base_ch, num_down, num_residual, res_norm='instance', down_norm='instance',input_ch= 3):
        super(Encoder2, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x, nce, layers=[0,1,2,6]):
        sides = [] 
        if nce: 

            for layer in self.layers:
                x = layer(x)
                sides.append(x)
        else: 
            for layers_id,layer in enumerate(self.layers):
                x = layer(x)
                if layers_id in layers:
                    sides.append(x)

        return sides[::-1] 

class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer', fuse=False,disen1=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_ch = input_ch*2
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)  

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch = input_ch//2
        m = ConvolutionBlock(
            in_channels=base_ch*2, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch*2)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                    nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))  
            self.fuse = [getattr(self, "fuse{}".format(i)) for i in range(num_sides)]

            self.fuse = lambda x, y, z, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y, z), 1))

        else:
            self.fuse = lambda x, y,z, i: x + y + z

    def forward(self, x, sides=[], sides_fre=[]):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            x = self.layers[i](x)

        for i, j in enumerate(range(m - n, m)):
            if i == 0:
                x = self.layers[j](x)
            elif i==5:
                x = x.float()
                x = self.fuse(x, sides[i], sides_fre[i], i) 
                x = self.layers[j](x) 
            else: 
                x = self.fuse(x, sides[i], sides_fre[i], i) 
                x = self.layers[j](x)

        return x

    def forward1(self, sides, sides_fre):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"
        for i in range(m - n, m):
            if i == 0:
                x_lower = torch.cat((sides[i], sides_fre[i]), 1)
                x = self.layers[i](x_lower)
            else:
                x = self.fuse(x, sides[i], sides_fre[i], i)
                x = self.layers[i](x)
        return x

class ENC_fea(nn.Module): 
    def __init__(self, base_ch=64, num_down=2, num_residual=3, res_norm='instance', down_norm='instance'):
        super(ENC_fea, self).__init__()
        self.enc_spa = Encoder1( base_ch, num_down, num_residual, res_norm, down_norm, input_ch=3 ) 
        self.enc_fre = Encoder1( base_ch, num_down, num_residual, res_norm, down_norm, input_ch=6 ) 
        self.fusion  = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=512)

    def forward(self,x):  
        x = x.float()
        x_fre = torch.fft.fft2( x, norm='backward' )  
        x_fre = torch.cat((x_fre.real, x_fre.imag), 1) 
     
        x_structure, side = self.enc_spa(x)
        x_fre, side_fre = self.enc_fre(x_fre)
       
        x_structure_fusion = self.fusion(x_structure, x_fre)

        return x_structure_fusion, side, side_fre 
    
class ENC_side(nn.Module): 
    def __init__(self, base_ch=64, num_down=2, num_residual=3,
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True):
        super(ENC_side, self).__init__()
        self.enc_spa = Encoder2( base_ch, num_down, num_residual, res_norm, down_norm, input_ch=3 ) 
        self.enc_fre = Encoder2( base_ch, num_down, num_residual, res_norm, down_norm, input_ch=6 ) 
    def forward(self,x,nce=True):
        x = x.float()
        x_fre = torch.fft.fft(x, norm='backward') 
        x_fre = torch.cat((x_fre.real, x_fre.imag), 1)
      
        sides = self.enc_spa(x,nce)
        sides_fre = self.enc_fre(x_fre,nce)

        return sides, sides_fre
    
class DEC(nn.Module): 
    def __init__(self, input_ch=3, base_ch=64, num_down=2, num_residual=3, num_sides="all",
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True,disen1=False):
        super(DEC, self).__init__()
        self.n = num_down + num_residual + 1  if num_sides == "all" else num_sides
        self.decoder = Decoder(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse) 

    def forward(self, x, side=[], side1=[], disen=False):
        if disen:
            x = self.decoder.forward1(x, side)
        else:
            if len(side) != 0:
                x = self.decoder(x, side, side1)
            else:
                x = self.decoder(x)

        return x
    
class Dual_Domain_GEN(nn.Module): 
    def __init__(self):
        super(Dual_Domain_GEN, self).__init__()

        self.enc_low  = ENC_fea() 
        self.enc_high = ENC_fea() 
        self.enc_art  = ENC_side()

        self.dec1 = DEC()  
        self.dec2 = DEC(disen1=True)   

    def forward1(self, x, y):
        side, side_fre = self.enc_art(x)  

        x_structure,_,_ = self.enc_low(x)  
        y_structure,_,_ = self.enc_high(y) 

        y_output = self.dec2(y_structure, side, side_fre) 
        x_output = self.dec1(x_structure)  

        return x_output, y_output, x_structure, y_structure, side

    def forward2(self, x, y): 
        side, side_fre = self.enc_art(x)  
        x_structure,_,_ = self.enc_low(x)
        y_structure,_,_ = self.enc_high(y)

        x_output = self.dec1(x_structure, side, side_fre)  
        y_output = self.dec2(y_structure) 

        return x_output, y_output

    def disen(self, x): 
        side, side_fre = self.enc_art(x)
        x = self.dec2(side, side_fre, disen=True)
        return x

if __name__ == '__main__':
    model = Dual_Domain_GEN()
    a     = torch.ones(1,1,256,256)
    a_fft = torch.ones(1,2,256,256)
    b     = torch.ones(1,1,256,256)
    b_fft = torch.ones(1,2,256,256)

    out2 = model.disen(a)
    print()
    print(out2.shape)
