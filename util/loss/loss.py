import torch

class MSELoss:
    def loss(self, ground_truth, pred_img):
        mse_loss = torch.mean((ground_truth - pred_img)**2)
        return mse_loss