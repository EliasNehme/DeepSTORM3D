# Import modules and libraries
import numpy as np
import torch
from torch.nn import Module, MaxPool3d, ConstantPad3d, MaxPool2d
from torch.nn.functional import conv3d


# convert gpu tensors to numpy
def tensor_to_np(x):
    return np.squeeze(x.cpu().numpy())


# post-processing on GPU: thresholding and local maxima finding
class Postprocess(Module):
    def __init__(self, setup_params, thresh=40, radius=4, keep_singlez=True):
        super().__init__()
        self.device = setup_params['device']
        self.psize_xy = setup_params['pixel_size_rec']
        self.psize_z = setup_params['pixel_size_axial']
        self.zmin = setup_params['zmin']
        self.upsampling_shift = 2  # 0 => due to floor(W/2) affected by upsampling factor of 4
        self.thresh = thresh
        self.r = radius
        self.keep_singlez = keep_singlez
        self.maxpool = MaxPool3d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        self.maxpool2 = MaxPool2d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

        # construct the local average filters
        filt_vec = np.arange(-self.r, self.r + 1)
        yfilter, zfilter, xfilter = np.meshgrid(filt_vec, filt_vec, filt_vec)
        xfilter = torch.FloatTensor(xfilter).unsqueeze(0).unsqueeze(0)
        yfilter = torch.FloatTensor(yfilter).unsqueeze(0).unsqueeze(0)
        zfilter = torch.FloatTensor(zfilter).unsqueeze(0).unsqueeze(0)
        sfilter = torch.ones_like(xfilter)
        self.local_filter = torch.cat((sfilter, xfilter, yfilter, zfilter), 0).to(self.device)

    def keep_maxz(self, conf_vol):

        # get the maximum value in z per xy
        D, H, W = conf_vol.shape
        max_proj, _ = torch.max(conf_vol, dim=0, keepdim=True)

        # keep only local maxima in 2d
        max_proj = self.maxpool2(max_proj.unsqueeze(0))
        max_proj = max_proj.squeeze(0)

        # keep only maximum
        conf_vol_out = torch.where(conf_vol == max_proj.expand(D, H, W), conf_vol, self.zero)

        return conf_vol_out
    
    def local_avg(self, xbool, ybool, zbool, pred_vol_pad, num_pts, device):

        # create the concatenated tensor of all local volumes
        pred_vol_all = torch.zeros(num_pts, 1, self.r*2 + 1, self.r*2 + 1, self.r*2 + 1).to(device)
        for pt in range(num_pts):

            # local 3D volume
            xpt = [xbool[pt], xbool[pt] + 2 * self.r + 1]
            ypt = [ybool[pt], ybool[pt] + 2 * self.r + 1]
            zpt = [zbool[pt], zbool[pt] + 2 * self.r + 1]
            pred_vol_all[pt, :] = pred_vol_pad[:, :, zpt[0]:zpt[1], ypt[0]:ypt[1], xpt[0]:xpt[1]]

        # convolve it using conv3d
        sums = conv3d(pred_vol_all, self.local_filter)

        # squeeze the sums and convert them to local perturbations
        xloc = sums[:, 1] / sums[:, 0]
        yloc = sums[:, 2] / sums[:, 0]
        zloc = sums[:, 3] / sums[:, 0]

        return xloc, yloc, zloc

    def forward(self, pred_vol):

        # check size of the prediction and expand it accordingly to be 5D
        num_dims = len(pred_vol.size())
        if np.not_equal(num_dims, 5):
            if num_dims == 4:
                pred_vol = pred_vol.unsqueeze(0)
            else:
                pred_vol = pred_vol.unsqueeze(0)
                pred_vol = pred_vol.unsqueeze(0)

        # apply the threshold
        pred_thresh = torch.where(pred_vol > self.thresh, pred_vol, self.zero)

        # apply the 3D maxpooling operation to find local maxima
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)
        
        # keep only a single z in each xy sub-pixel
        if self.keep_singlez:
            conf_vol = torch.squeeze(conf_vol)
            conf_vol = self.keep_maxz(conf_vol)

        # find locations of confs (bigger than 0)
        conf_vol = torch.squeeze(conf_vol)
        batch_indices = torch.nonzero(conf_vol)
        zbool, ybool, xbool = batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]

        # if the prediction is empty return None otherwise convert to list of locations
        if len(zbool) == 0:
            xyz_rec = None
            conf_rec = None

        else:

            # pad the result with radius_px 0's for average calc.
            pred_vol_pad = self.pad(pred_vol)

            # for each point calculate local weighted average
            num_pts = len(zbool)
            xloc, yloc, zloc = self.local_avg(xbool, ybool, zbool, pred_vol_pad, num_pts, self.device)

            # convert lists and tensors to numpy
            xloc, yloc, zloc = tensor_to_np(xloc), tensor_to_np(yloc), tensor_to_np(zloc)
            xbool, ybool, zbool = tensor_to_np(xbool), tensor_to_np(ybool), tensor_to_np(zbool)

            # dimensions of the prediction
            D, H, W = conf_vol.size()

            # calculate the recovered positions assuming mid-voxel
            xrec = (xbool + xloc - np.floor(W / 2) + self.upsampling_shift + 0.5) * self.psize_xy
            yrec = (ybool + yloc - np.floor(H / 2) + self.upsampling_shift + 0.5) * self.psize_xy
            zrec = (zbool + zloc + 0.5) * self.psize_z + self.zmin

            # rearrange the result into a Nx3 array
            xyz_rec = np.column_stack((xrec, yrec, zrec))

            # confidence of these positions
            conf_rec = conf_vol[zbool, ybool, xbool]
            conf_rec = tensor_to_np(conf_rec)

        return xyz_rec, conf_rec
