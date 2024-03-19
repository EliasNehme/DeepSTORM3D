# Import modules and libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================================================================
# Overlap measure + KDE loss in 3D
# ======================================================================================================================


def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)


    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


# create a 3D gaussian kernel
def GaussianKernel(shape=(7, 7, 7), sigma=1, normfactor=1):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]) in 3D
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    """
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        h = h * normfactor
    """
    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor
    h = torch.from_numpy(h).type(torch.FloatTensor)
    h = h.unsqueeze(0)
    h = h.unsqueeze(1)
    return h


# define the 3D extended loss function from DeepSTORM
class KDE_loss3D(nn.Module):
    def __init__(self, factor, device):
        super(KDE_loss3D, self).__init__()
        self.kernel = GaussianKernel().to(device)
        self.factor = factor

    def forward(self, pred_bol, target_bol):

        # extract kernel dimensions
        N, C, D, H, W = self.kernel.size()
        
        # extend prediction and target to have a single channel
        target_bol = target_bol.unsqueeze(1)
        pred_bol = pred_bol.unsqueeze(1)

        # KDE for both input and ground truth spikes
        Din = F.conv3d(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Dtar = F.conv3d(target_bol, self.factor*self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

        # kde loss
        kde_loss = nn.MSELoss()(Din, Dtar)
        
        # final loss
        final_loss = kde_loss + dice_loss(pred_bol/self.factor, target_bol)

        return final_loss


# ======================================================================================================================
# Jaccard index
# ======================================================================================================================


# calculates the jaccard coefficient approximation using per-voxel probabilities
def jaccard_coeff(pred, target):
    """
    jaccard index = TP / (TP + FP + FN)
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # smoothing parameter
    smooth = 1e-6
    
    # number of examples in the batch
    N = pred.size(0)

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(N,-1)
    tflat = target.contiguous().view(N,-1)
    intersection = (iflat * tflat).sum(1)
    jacc_index = (intersection / (iflat.sum(1) + tflat.sum(1) - intersection + smooth)).mean()

    return jacc_index
