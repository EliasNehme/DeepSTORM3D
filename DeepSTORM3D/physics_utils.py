# Import modules and libraries
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from math import pi


# ======================================================================================================================
# back focal plane (Fourier plane) grids calculation for efficient on the fly image simulation given emitter positions
# and a phase mask
# ======================================================================================================================


# function calculates back focal plane grids necessary to calculate PSF using
# the pupil function approach
def calc_bfp_grids(setup_params):

    # extract light and sensor properties
    lamda, NA, M, f_4f = setup_params['lamda'], setup_params['NA'], setup_params['M'], setup_params['f_4f']
    pixel_size_CCD, pixel_size_SLM = setup_params['pixel_size_CCD'], setup_params['pixel_size_SLM']

    # sample properties
    noil, nwater = setup_params['noil'], setup_params['nwater']

    # mask and focus properties
    Hmask, Wmask, NFP = setup_params['Hmask'], setup_params['Wmask'], setup_params['NFP']

    # SLM, and back focal plane properties
    BFPdiam_um = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # [um]
    N = Wmask  # number of pixels in the SLM
    px_per_um = 1 / pixel_size_SLM  # [1/um]
    BFPdiam_px = BFPdiam_um * px_per_um  # BFP diameter in pixels
    CCD_size_px = lamda * f_4f * px_per_um / pixel_size_CCD  # final size of FFT on camera
    pad_px = int(np.round(0.5 * (CCD_size_px - N)))  # needed padding before and after phase mask in FP
    if pad_px != 0:
        print('Padding needed in forward model to match SLM and CCD pixel-size!')

    # cartesian and polar grid in BFP
    Xphys = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) / px_per_um
    XI, ETA = np.meshgrid(Xphys, Xphys)  # cartesian physical coords. in SLM space [um]
    Xang = np.linspace(-1, 1, N) * NA / noil * N / BFPdiam_px  # each pixel is NA/(BFPdiam_px/2*n1) wide
    XX, YY = np.meshgrid(Xang, Xang)
    r = np.sqrt(XX ** 2 + YY ** 2)  # radial coordinate s.t. r = NA/noil at edge of E field support

    # defocus aberration in oil
    koil = 2 * pi * noil / lamda  # [1/um]
    sin_theta_oil = r  # um
    circ_oil = sin_theta_oil < NA / noil
    circ_oil = circ_oil.astype("float32")
    cos_theta_oil = np.sqrt(1 - (sin_theta_oil * circ_oil) ** 2) * circ_oil
    defocusAberr = np.exp(1j * koil * (-NFP) * cos_theta_oil)

    # create the circular aperture
    kwater = 2 * pi * nwater / lamda
    sin_theta_water = noil / nwater * sin_theta_oil
    circ_water = sin_theta_water < 1
    circ_water = circ_water.astype("float32")
    sin_theta_water = sin_theta_water * circ_water
    cos_theta_water = np.sqrt(1 - sin_theta_water ** 2) * circ_water

    # circular aperture to impose on the mask in the SLM
    circ = circ_water.astype("float32")

    # Inputs for the calculation of lateral and axial phases
    Xgrid = 2 * pi * XI * M / (lamda * f_4f)
    Ygrid = 2 * pi * ETA * M / (lamda * f_4f)
    Zgrid = kwater * cos_theta_water

    # defocus term for psf visualization later throughout training
    zmin, zmax = setup_params['zmin'], setup_params['zmax']
    zvis = np.linspace(zmin, zmax + 1, 6)  # in [um]
    zvis3D = np.transpose(np.tile(zvis, [Hmask, Wmask, 1]), (2, 0, 1))
    defocus = circ_water * np.exp(1j * zvis3D * Zgrid)
    vis_term = defocus * defocusAberr

    # save grids to the setup param dictionary
    setup_params['zvis'], setup_params['vis_term'], setup_params['circ'],  = zvis, vis_term, circ
    setup_params['Xgrid'], setup_params['Ygrid'], setup_params['Zgrid'] = Xgrid, Ygrid, Zgrid
    setup_params['defocusAberr'] = defocusAberr

    return setup_params


# ======================================================================================================================
# Conversion of xyz to phases in 2D Fourier space according to the pre-calculated terms
# ======================================================================================================================


# function calculates the phases in fourier space for given emitter positions
def EmittersToPhases(xyz, setup_params):
    
    # extract Xg for spatial dimensions calculation
    Xg = setup_params['Xgrid']
    H, W = Xg.shape
    
    # number of emitters and examples in batch
    Nbatch, Nemitters, Ndims = xyz.shape
    
    # initialize emitter specific constant
    phase_emitter_location = np.zeros((Nbatch,Nemitters,H,W), dtype=complex) 
    
    # loop over the number of samples in the batch
    for sample in range(Nbatch):

        # loop over all emitters and generate their PSF image
        for emitter in range(Nemitters):

            # current emitter location in [m]
            x0, y0, z0 = xyz[sample, emitter, :]

            # phase due to lateral shift (x0,y0)
            phase_lat = np.exp(1j * (x0 * setup_params['Xgrid'] + y0 * setup_params['Ygrid']))

            # phase due to axial shift z0
            phase_ax = np.exp(1j * z0 * setup_params['Zgrid'])

            # emitter-location induced phase constant
            phase_emitter_location[sample, emitter, :, :] = phase_lat * phase_ax
    
    # total constant phase due to emitter location and general defocus aberration
    phase_emitter = phase_emitter_location * setup_params['defocusAberr'] * setup_params['circ'] 
    
    return phase_emitter


# ======================================================================================================================
# fftshift and ifftshift batch implementations in torch
# ======================================================================================================================


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


# ======================================================================================================================
# Phase to psf conversion function:
# function takes the phase mask, emitter phases in fourier space, and emitted photons per emitter; and outputs a set
# of images (one per emitter) depending on the 3D location
# ======================================================================================================================


class MaskPhasesToPSFs(Function):

    # forward path
    @staticmethod
    def forward(ctx, mask, phase_emitter, Nphotons, device):

        # number of emitters
        Nbatch, Nemitters, H, W, ri = phase_emitter.size()

        # repeat mask to the 3rd and 4th dim.
        phasemask4D = mask.expand(Nbatch, Nemitters, H, W)
        
        # phase term due to phase mask
        phasemask_term = torch.zeros((Nbatch, Nemitters, H, W, ri)).to(device)
        phasemask_term[:,:,:,:,0] = torch.cos(phasemask4D)
        phasemask_term[:,:,:,:,1] = torch.sin(phasemask4D)

        # total phase term
        phase4D = torch.zeros((Nbatch, Nemitters, H, W, ri)).to(device)
        phase4D[:,:,:,:,0] = phasemask_term[:,:,:,:,0]*phase_emitter[:,:,:,:,0] - phasemask_term[:,:,:,:,1]*phase_emitter[:,:,:,:,1]
        phase4D[:,:,:,:,1] = phasemask_term[:,:,:,:,0]*phase_emitter[:,:,:,:,1] + phasemask_term[:,:,:,:,1]*phase_emitter[:,:,:,:,0]
        
        # calculate the centered fourier transform on the H, W dims
        fft_res = batch_fftshift2d(torch.fft(batch_ifftshift2d(phase4D), 2, True))

        # take the absolute value squared
        fft_abs_square = torch.sum(fft_res**2, 4)
        
        # calculate normalization factor
        sumhw = fft_abs_square.sum(2).sum(2)
        normfactor = Nphotons/sumhw

        # depth-wise normalization factor
        images_norm = torch.zeros((Nbatch, Nemitters, H, W)).to(device)
        for i in range(Nbatch):
            for j in range(Nemitters):
                images_norm[i, j, :, :] = fft_abs_square[i, j, :, :] * normfactor[i, j]
        
        # save tensors for backward pass
        ctx.device, ctx.fft_res, ctx.phasemask_term, ctx.phase_emitter = device, fft_res, phasemask_term, phase_emitter
        ctx.normfactor, ctx.Nbatch, ctx.Nemitters, ctx.H, ctx.W = normfactor, Nbatch, Nemitters, H, W
        
        return images_norm

    # backward pass
    @staticmethod
    def backward(ctx, grad_output):

        # extract saved tensors for gradient update
        device, fft_res, phasemask_term, phase_emitter = ctx.device, ctx.fft_res, ctx.phasemask_term, ctx.phase_emitter
        normfactor, Nbatch, Nemitters, H, W = ctx.normfactor, ctx.Nbatch, ctx.Nemitters, ctx.H, ctx.W

        # gradient w.r.t the single-emitter images
        grad_input = grad_output.data

        # depth-wise normalization factor
        for i in range(Nbatch):
            for j in range(Nemitters):
                grad_input[i, j, :, :] = grad_input[i, j, :, :] * normfactor[i, j]

        # gradient of abs squared
        grad_abs_square = torch.zeros((Nbatch, Nemitters, H, W, 2)).to(device)
        grad_abs_square[:, :, :, :, 0] = 2*grad_input*fft_res[:, :, :, :, 0]
        grad_abs_square[:, :, :, :, 1] = 2*grad_input*fft_res[:, :, :, :, 1]

        # calculate the centered inverse fourier transform on the H, W dims
        grad_fft = batch_fftshift2d(torch.ifft(batch_ifftshift2d(grad_abs_square), 2, True))
        
        # gradient w.r.t phase mask phase_term
        grad_phasemask_term = torch.zeros((Nbatch, Nemitters, H, W, 2)).to(device)
        grad_phasemask_term[:, :, :, :, 0] = grad_fft[:, :, :, :, 0]*phase_emitter[:, :, :, :, 0] + grad_fft[:, :, :, :, 1]*phase_emitter[:, :, :, :, 1]
        grad_phasemask_term[:, :, :, :, 1] = -grad_fft[:, :, :, :, 0]*phase_emitter[:, :, :, :, 1] + grad_fft[:, :, :, :, 1]*phase_emitter[:, :, :, :, 0]

        # gradient w.r.t the phasemask 4D
        grad_phasemask4D = -grad_phasemask_term[:, :, :, :, 0]*phasemask_term[:, :, :, :, 1] + grad_phasemask_term[:, :, :, :, 1]*phasemask_term[:, :, :, :, 0] 
        
        # sum to get the final gradient
        grad_phasemask = grad_phasemask4D.sum(0).sum(0)

        return grad_phasemask, None, None, None


# ======================================================================================================================
# blurring module: blur slightly each emitter using a different gaussian
# ======================================================================================================================


# This function creates a 2D gaussian filter with std=1, without normalization.
# during training this filter is scaled with a random std to simulate different blur per emitter
def gaussian2D_unnormalized(shape=(7, 7), sigma=1.0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    hV = torch.from_numpy(h).type(torch.FloatTensor)
    return hV


# Blur layer:
# this layer blurs each emitter with a slightly different Gaussian  with std in [0.5,1.5] to include a low-pass effect
class BlurLayer(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.device = setup_params['device']
        self.gauss = gaussian2D_unnormalized(shape=(9, 9)).to(self.device)
        self.std_min = setup_params['blur_std_range'][0]
        self.std_max = setup_params['blur_std_range'][1]

    def forward(self, PSFs):

        # number of the input PSF images
        Nemitters = PSFs.size(1)

        # generate random gaussian blur for each emitter
        RepeatedGaussian = self.gauss.expand(1, Nemitters, 9, 9)
        stds = (self.std_min + (self.std_max - self.std_min) * torch.rand((Nemitters, 1))).to(self.device)
        MultipleGaussians = torch.zeros_like(RepeatedGaussian)
        for i in range(Nemitters):
            MultipleGaussians[:, i, :, :] = 1 / (2 * pi * stds[i] ** 2) * torch.pow(RepeatedGaussian[:, i, :, :],
                                                                                    1 / (stds[i] ** 2))

        # blur each emitter with slightly different gaussian
        images4D_blur = F.conv2d(PSFs, MultipleGaussians, padding=(2, 2))

        # result
        return images4D_blur


# ======================================================================================================================
# non-uniform background noise layer:
# this layer takes in an input and add a random super-gaussian non-uniform background to model the cell/nucleus
# auto-fluorescence
# ======================================================================================================================


class NonUniformBg(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.batch_size_gen, self.H, self.W = setup_params['batch_size_gen'], setup_params['H'], setup_params['W']
        m, n = [(ss - 1.) / 2. for ss in (self.H, self.W)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        self.Xbg = torch.from_numpy(x).type(torch.FloatTensor)
        self.Ybg = torch.from_numpy(y).type(torch.FloatTensor)
        self.offsetX = setup_params['nonunif_bg_offset'][0]
        self.offsetY = setup_params['nonunif_bg_offset'][1]
        self.angle_min = setup_params['nonunif_bg_theta_range'][0]
        self.angle_max = setup_params['nonunif_bg_theta_range'][1]
        self.bmin = setup_params['nonunif_bg_minvals'][0]
        self.bmax = setup_params['nonunif_bg_minvals'][1]
        self.device = setup_params['device']

    def forward(self, input):

        # generate different non-uniform backgrounds
        Nbatch = input.size(0)
        bgs = torch.zeros((Nbatch, 1, self.H, self.W))

        for i in range(self.batch_size_gen):

            # cast a new center in the range [+-offsetX ,+-offsetY]
            x0 = -self.offsetX + torch.rand(1) * self.offsetX * 2
            y0 = -self.offsetY + torch.rand(1) * self.offsetY * 2

            # cast two stds
            sigmax = self.W / 5 + torch.rand(1) * self.W/5
            sigmay = self.H / 5 + torch.rand(1) * self.H/5

            # cast a new angle
            theta = self.angle_min + torch.rand(1) * (self.angle_max - self.angle_min)

            # calculate rotated gaussian coefficients
            a = torch.cos(theta) ** 2 / (2 * sigmax ** 2) + torch.sin(theta) ** 2 / (2 * sigmay ** 2)
            b = -torch.sin(2 * theta) / (4 * sigmax ** 2) + torch.sin(2 * theta) / (4 * sigmay ** 2)
            c = torch.sin(theta) ** 2 / (2 * sigmax ** 2) + torch.cos(theta) ** 2 / (2 * sigmay ** 2)

            # minimal and maximal background
            bmin = self.bmin + torch.rand(1)*self.bmin/2
            bmax = self.bmax + torch.rand(1)*self.bmax/2

            # calculate rotated gaussian and scale it
            h = torch.exp(-(a * (self.Xbg - x0) ** 2 + 2 * b * (self.Xbg - x0) * (self.Ybg - y0) + c * (self.Ybg - y0) ** 2) ** 2)
            maxh = h.max()
            h[h < 1e-6 * maxh] = 0
            minh = h.min()
            h = (h - minh) / (maxh - minh) * (bmax - bmin) + bmin
            h = h.type(torch.FloatTensor)

            # resulting non-uniform bg
            bgs[i, :, :, :] = h

        return input + bgs.to(self.device)


# ======================================================================================================================
# Poisson noise layer implemented in numpy (not differentiable)
# ======================================================================================================================


def poisson_noise_numpy(input, setup_params):

    # cast input to numpy
    input_numpy = input.cpu().numpy()

    # number of images
    Nbatch = input_numpy.shape[0]

    # apply poison noise in numpy for each sample
    out = torch.zeros((Nbatch, 1, setup_params['H'], setup_params['W']))
    for i in range(Nbatch):
        out_np = np.random.poisson(input_numpy[i, :, :, :], size=None)
        out[i, :, :, :] = torch.from_numpy(out_np).type(torch.FloatTensor)

    return out.to(setup_params['device'])


# ======================================================================================================================
# Differentiable approximation of the Poisson noise layer:
# this layer takes in an input and approximates the Poisson noise using the reparametrization trick from VAEs
# ======================================================================================================================


class poisson_noise_approx(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.H, self.W = setup_params['H'], setup_params['W']
        self.device = setup_params['device']

    def forward(self, input):

        # number of images
        Nbatch = input.size(0)

        # approximate the poisson noise using CLT and reparameterization
        input_poiss = input + torch.sqrt(input)*torch.randn(Nbatch, 1, self.H, self.W).type(torch.FloatTensor).to(self.device)

        # result
        return input_poiss


# ======================================================================================================================
# Gaussian read-out noise layer:
# this layer adds read noise with a uniform baseline and a non-uniform standard deviation modelled using a random
# super-gaussian.
# ======================================================================================================================


class ReadNoise(nn.Module):
    def __init__(self, setup_params):
        super().__init__()

        # read noise standard deviation spatially distrbution
        self.std_nonunif = setup_params['read_noise_nonuinf']
        self.batch_size_gen, self.H, self.W = setup_params['batch_size_gen'], setup_params['H'], setup_params['W']
        if self.std_nonunif:
            nonunif_bg_minvals = setup_params['nonunif_bg_minvals']
            setup_params['nonunif_bg_minvals'] = [0, 1]
            self.non_uniform_bg = NonUniformBg(setup_params)
            setup_params['nonunif_bg_minvals'] = nonunif_bg_minvals
        self.device = setup_params['device']
        self.sigma_read_min = torch.Tensor([setup_params['read_noise_std_range'][0]]).to(self.device)
        self.sigma_read_max = torch.Tensor([setup_params['read_noise_std_range'][1]]).to(self.device)
        self.baseline_min = torch.Tensor([setup_params['read_noise_baseline_range'][0]]).to(self.device)
        self.baseline_max = torch.Tensor([setup_params['read_noise_baseline_range'][1]]).to(self.device)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

    def forward(self, input):

        # number of images
        Nbatch = input.size(0)

        # cast a random baseline
        ones_tensor = torch.ones((Nbatch, 1, self.H, self.W)).to(self.device)
        baseline = ones_tensor*torch.rand(1).to(self.device)*(self.baseline_max - self.baseline_min) + self.baseline_min

        # decide whether the standard deviation is uniform or non-uniform across the FOV
        if self.std_nonunif:

            # generate a spatial distribution for the standard deviation
            zeros_tensor = torch.zeros((Nbatch, 1, self.H, self.W)).to(self.device)
            bg_01 = self.non_uniform_bg(zeros_tensor)

            # scale non-uniform background to create a standard deviation map in the FOV
            bg_std = bg_01*(self.sigma_read_max - self.sigma_read_min) + self.sigma_read_min

        else:

            # uniform std across the FOV
            bg_std = ones_tensor * (self.sigma_read_max - self.sigma_read_min) + self.sigma_read_min

        # resulting noise map
        read_noise = baseline + bg_std*torch.randn((Nbatch, 1, self.H, self.W)).type(torch.FloatTensor).to(self.device)

        return torch.max(input + read_noise, self.zero)


# ======================================================================================================================
# Overall noise layer:
# this layer implements the acceptable noise model for SMLM imaging: Poisson + Gaussian
# ======================================================================================================================


class NoiseLayer(nn.Module):
    def __init__(self, setup_params):
        super().__init__()

        # if the mask is being learned approximate poisson noise
        self.learn_mask = setup_params['learn_mask']
        if self.learn_mask:
            self.poiss = poisson_noise_approx(setup_params)
        self.nonunif_bg_flag = setup_params['nonunif_bg_flag']
        if self.nonunif_bg_flag:
            self.non_uniform_bg = NonUniformBg(setup_params)
        else:
            self.unif_bg = setup_params['unif_bg']
        self.read_noise_flag = setup_params['read_noise_flag']
        if self.read_noise_flag:
            self.read_noise = ReadNoise(setup_params)
        self.setup_params = setup_params

    def forward(self, input):

        # add a uniform/non-uniform background
        if self.nonunif_bg_flag:
            inputb = self.non_uniform_bg(input)
        else:
            inputb = input + self.unif_bg

        # apply poisson noise (approximate or non-differentiable)
        if self.learn_mask:
            inputb_poiss = self.poiss(inputb)
        else:
            inputb_poiss = poisson_noise_numpy(inputb, self.setup_params)

        # apply uniform/non-uniform read noise if specified
        if self.read_noise_flag:
            input_poiss_read = self.read_noise(inputb_poiss)
        else:
            input_poiss_read = inputb_poiss

        # result
        return input_poiss_read


# ======================================================================================================================
# Cropping layer: keeps only the center part of the FOV to prevent unnecessary processing
# ======================================================================================================================


class Croplayer(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.H, self.W = setup_params['H'], setup_params['W']

    def forward(self, images4D):

        # size of the PSFs array
        Nbatch, C, Him, Wim = images4D.size()

        # crop the central portion in the xy plane to remove redundant zeros
        Xlow = int(np.floor(Wim / 2) - np.floor(self.W / 2))
        Xhigh = int(np.floor(Wim / 2) + np.floor(self.W / 2) + 1)
        Ylow = int(np.floor(Him / 2) - np.floor(self.H / 2))
        Yhigh = int(np.floor(Him / 2) + np.floor(self.H / 2) + 1)
        images4D_crop = images4D[:, :, Ylow:Yhigh, Xlow:Xhigh].contiguous()

        return images4D_crop


# ======================================================================================================================
# Normalization layer: takes the resulting images and project them into the interval [0,1] to prevent scale ambiguity
# ======================================================================================================================


class Normalize01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result_noisy):

        # number of samples in each batch
        Nbatch = result_noisy.size(0)

        # [0,1] normalization to prevent scaling volnurability
        result_noisy_01 = torch.zeros_like(result_noisy)
        for i in range(Nbatch):
            min_val = (result_noisy[i, :, :, :]).min()
            max_val = (result_noisy[i, :, :, :]).max()
            result_noisy_01[i, :, :, :] = (result_noisy[i, :, :, :] - min_val) / (max_val - min_val)

        return result_noisy_01


# ======================================================================================================================
# Physical encoding layer, from 3D to 2D:
# this layer takes in the learnable parameter "mask", and the input continuous locations as phases in fourier space,
# and output the resulting 2D image corresponding to the emitters location.
# ======================================================================================================================


class PhysicalLayer(nn.Module):
    def __init__(self, setup_params):
        super(PhysicalLayer, self).__init__()
        self.device = setup_params['device']
        self.blur = BlurLayer(setup_params)
        self.crop = Croplayer(setup_params)
        self.noise = NoiseLayer(setup_params)
        self.norm_flag = setup_params['project_01']
        if self.norm_flag:
            self.norm01 = Normalize01()

    def forward(self, mask, phase_emitter, nphotons):

        # generate the PSF images from the phase mask and xyz locations
        PSF4D = MaskPhasesToPSFs.apply(mask, phase_emitter, nphotons, self.device)

        # blur each emitter with slightly different gaussian
        images4D_blur = self.blur(PSF4D)

        # crop relevant FOV area
        images4D_blur_crop = self.crop(images4D_blur)

        # apply the measurement noise model
        result_noisy = self.noise(images4D_blur_crop)

        # [0,1] normalization to prevent scaling volnurability
        if self.norm_flag:
            result_noisy_01 = self.norm01(result_noisy)
        else:
            result_noisy_01 = result_noisy

        return result_noisy_01
