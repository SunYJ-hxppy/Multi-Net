import torch
from monai.networks import one_hot

def OnehotNcombine( input ):    
    
    _, idx = torch.max( input, 1 )
    idx1 = idx.unsqueeze(1).repeat(1, input.size(1), 1, 1)
    input_onehot = torch.zeros_like( input )
    input_onehot.scatter_(1, idx1, 1)
    
    BG  = input_onehot[:,0,:,:]
    CSF = input_onehot[:,1,:,:]
    GM  = input_onehot[:,2,:,:]
    WM  = input_onehot[:,3,:,:]

    
    BG[ BG == 1 ] = 0  
    CSF[ CSF == 1 ] = 1
    GM[ GM == 1 ] = 2
    WM[ WM == 1 ] = 3
    
    output = BG + CSF + GM + WM    
    
    return output


def OnehotEncoding( input ):
    
    _, idx = torch.max( input, dim=1 )
    idx1 = idx.unsqueeze(1).repeat(1, input.size(1), 1, 1)
    input_onehot = torch.zeros_like( input )
    input_onehot.scatter_(1, idx1, 1)
    
    return input_onehot

def Combine( input ):
   
    BG  = input[:,0,:,:]
    CSF = input[:,1,:,:]
    GM  = input[:,2,:,:]
    WM  = input[:,3,:,:]
    
    BG[ BG == 1 ]   = 0  
    CSF[ CSF == 1 ] = 1
    GM[ GM == 1 ]   = 2
    WM[ WM == 1 ]   = 3
        
    output = BG + CSF + GM + WM
    
    return output


def OnehotNcombine3D( input, include_background=True ):    
    
    if include_background:
        
        _, idx = torch.max( input, 1 )                              
        idx1 = idx.unsqueeze(1).repeat(1, input.size(1), 1, 1, 1)   
        input_onehot = torch.zeros_like( input )
        input_onehot.scatter_(dim=1, index=idx1, value=1)           

        BG  = input_onehot[:,0,:,:,:]
        CSF = input_onehot[:,1,:,:,:]
        GM  = input_onehot[:,2,:,:,:]
        WM  = input_onehot[:,3,:,:,:]
        
        BG[ BG == 1 ] = 0 
        CSF[ CSF == 1 ] = 1
        GM[ GM == 1 ] = 2
        WM[ WM == 1 ] = 3
        
        output = BG + CSF + GM + WM    
    
    return output

def Combine3D( input ):
    
    BG  = input[:,0,:,:,:]
    CSF = input[:,1,:,:,:]
    GM  = input[:,2,:,:,:]
    WM  = input[:,3,:,:,:]
    
    BG[ BG == 1 ] = 0  
    CSF[ CSF == 1 ] = 1
    GM[ GM == 1 ] = 2
    WM[ WM == 1 ] = 3
        
    output = BG + CSF + GM + WM
    
    return output


def MinMaxNorm( input ):

    output = ( input - input.min() ) / ( input.max() - input.min() )

    return output

