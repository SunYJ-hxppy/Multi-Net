import torch
import torch.nn as nn


class GradientSmoothing(nn.Module):
    def __init__(self, energy_type):
        super(GradientSmoothing, self).__init__()
        self.energy_type = energy_type

    def forward(self, dvf):

        def gradient_dx(fv): 
            return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

        def gradient_dy(fv):
            return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

        def gradient_txy(Txyz, fn): 
            return torch.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], dim=3)

        def compute_gradient_norm(displacement, flag_l1=False): 
            dTdx = gradient_txy(displacement, gradient_dx) 
            dTdy = gradient_txy(displacement, gradient_dy) 

            if flag_l1:
                norms = torch.abs(dTdx) + torch.abs(dTdy) 
            else: 
                norms = dTdx ** 2 + dTdy ** 2 
            return torch.mean(norms)
        
        def compute_bending_energy(displacement):
            dTdx = gradient_txy(displacement, gradient_dx) 
            dTdy = gradient_txy(displacement, gradient_dy)
            dTdxx = gradient_txy(dTdx, gradient_dx)
            dTdyy = gradient_txy(dTdy, gradient_dy)
            dTdxy = gradient_txy(dTdx, gradient_dy)
            
            return torch.mean( dTdxx ** 2 + dTdyy ** 2 + 2 * dTdxy ** 2 ) 

        if self.energy_type == 'bending':
            energy = compute_bending_energy(dvf)
        elif self.energy_type == 'gradient-l2':
            energy = compute_gradient_norm(dvf)
        elif self.energy_type == 'gradient-l1':
            energy = compute_gradient_norm(dvf, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')

        return energy


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None): 
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y): 
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r) 

        return df

    def loss(self, _, y_pred): 
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()