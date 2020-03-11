# Import modules and libraries
import torch
import numpy as np
from skimage.io import imread
from collections import OrderedDict


# ======================================================================================================================
# saving and resuming utils
# ======================================================================================================================


# checkpoint saver for model weights and optimization status
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# resume training from a check point
def resume_from_checkpoint(model, optimizer, filepath):
    print("=> loading checkpoint to resume training")
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return start_epoch


# load pretrained LR model to fine-tune for HR recovery
def prone_state_dict(saved_state_dict, load_last=True):
    
    # go over saved keys
    saved_dict_short = {}
    for k, v in saved_state_dict.items():
        
        # check the seventh layer number of input channels
        if (k == 'layer7.conv.weight' and v.size(1) == 65) or not(load_last):
            break
        else:
            saved_dict_short[k] = v
    
    return OrderedDict(saved_state_dict.items())


# transform xyz from microns to nms for saving and handling later in ThunderSTORM
def xyz_to_nm(xyz_um, ch, cw, psize_rec_xy, zmin):

    xnm = (xyz_um[:, 0] + cw * psize_rec_xy) * 1000
    ynm = (xyz_um[:, 1] + ch * psize_rec_xy) * 1000
    znm = (xyz_um[:, 2] - zmin) * 1000

    return np.column_stack((xnm, ynm, znm))


# ======================================================================================================================
# Normalization factors calculation and image projection to the range [0,1]
# ======================================================================================================================


def normalize_01(im):
    return (im - im.min())/(im.max() - im.min())


def CalcMeanStd_All(path_train, labels):
    """
    function calculates the mean and std (per-pixel!) for the training dataset,
    both these normalization factors are used for training and validation.
    """
    num_examples = len(labels)
    mean = 0.0
    for i in range(num_examples):
        im_name_tiff = path_train + 'im' + str(i) + '.tiff'
        im_tiff = imread(im_name_tiff)
        mean += im_tiff.mean()
    mean = mean / num_examples
    var = 0.0
    for i in range(num_examples):
        im_name_tiff = path_train + 'im' + str(i) + '.tiff'
        im_tiff = imread(im_name_tiff)
        var += ((im_tiff - mean)**2).sum()
    H, W = im_tiff.shape
    var = var/(num_examples*H*W)
    std = np.sqrt(var)
    return mean, std

