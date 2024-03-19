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

    # the mask dictated by the objective NA
    circ_NA = r <= 1
    
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
    circ = circ_water.astype("float32") * circ_NA.astype("float32") * circ_oil.astype("float32")

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

# complex exponential of a real tensor, with added 2 dimensions at the end
class exp_complex(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.device = setup_params['device']

    def forward(self, r):
        N, C, H, W = r.size()
        exp_1jr = torch.zeros((N, C, H, W, 2)).to(self.device)
        exp_1jr[:, :, :, :, 0] = torch.cos(r)
        exp_1jr[:, :, :, :, 1] = torch.sin(r)
        return exp_1jr


# multiplication of two complex numbers with 2 channels tensors
# z1 times z2 assuming complex tensors
class multiply_complex(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.device = setup_params['device']

    def forward(self, z1, z2):
        z1_x_z2 = torch.zeros_like(z1).to(self.device)
        z1_x_z2[:, :, :, :, 0] = z1[:, :, :, :, 0] * z2[:, :, :, :, 0] - z1[:, :, :, :, 1] * z2[:, :, :, :, 1]
        z1_x_z2[:, :, :, :, 1] = z1[:, :, :, :, 0] * z2[:, :, :, :, 1] + z1[:, :, :, :, 1] * z2[:, :, :, :, 0]
        return z1_x_z2


# abs squared of a complex tensor
class abs_squared_complex(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return torch.sum(z ** 2, -1)


# fft of complex tensor using the new function
class fft_complex(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z2 = batch_ifftshift2d(z)
        z3 = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(z2), dim=(2, 3), norm="ortho"))
        return batch_fftshift2d(z3)


# normalizing image by number of photons
class normalize_photons(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.device = setup_params['device']

    def forward(self, images, nphotons):
        sumhw = images.sum(2).sum(2)
        normfactor = nphotons / sumhw
        images_norm = torch.zeros_like(images).to(self.device)
        for i in range(images_norm.size(0)):
            for j in range(images_norm.size(1)):
                images_norm[i, j, :, :] = images[i, j, :, :] * normfactor[i, j]
        return images_norm


class MaskPhasesToPSFs(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.device = setup_params['device']
        self.exp_complex = exp_complex(setup_params)
        self.multiply_complex = multiply_complex(setup_params)
        self.abs_squared_complex = abs_squared_complex()
        self.fft_complex = fft_complex()
        self.normalize_photons = normalize_photons(setup_params)

    # forward path
    def forward(self, mask, phase_emitter, Nphotons):

        # number of emitters
        Nbatch, Nemitters, H, W, ri = phase_emitter.size()

        # repeat mask to the 3rd and 4th dim.
        phasemask4D = mask.expand(Nbatch, Nemitters, H, W)
        
        # phase term due to phase mask
        phasemask_term = self.exp_complex(phasemask4D)

        # total phase term
        phase4D = self.multiply_complex(phasemask_term, phase_emitter)
        
        # calculate the centered fourier transform on the H, W dims
        fft_res = self.fft_complex(phase4D)

        # take the absolute value squared
        fft_abs_square = self.abs_squared_complex(fft_res)
        
        # normalize image with the number of photons
        images_norm = self.normalize_photons(fft_abs_square, Nphotons)
        
        return images_norm


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
        images4D_blur = F.conv2d(PSFs, MultipleGaussians, padding=(4, 4))

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
        self.mask = MaskPhasesToPSFs(setup_params)
        self.blur = BlurLayer(setup_params)
        self.crop = Croplayer(setup_params)
        self.noise = NoiseLayer(setup_params)
        self.norm_flag = setup_params['project_01']
        if self.norm_flag:
            self.norm01 = Normalize01()

    def forward(self, mask, phase_emitter, nphotons):
        
        # generate the PSF images from the phase mask, xyz locations, and orientation params
        PSF4D = self.mask(mask, phase_emitter, nphotons)
        
        # crop relevant FOV area
        images4D_crop = self.crop(PSF4D)

        # blur each emitter with slightly different gaussian
        images4D_crop_blur = self.blur(images4D_crop)

        # apply the measurement noise model
        result_noisy = self.noise(images4D_crop_blur)

        # [0,1] normalization to prevent scaling volnurability
        if self.norm_flag:
            result_noisy_01 = self.norm01(result_noisy)
        else:
            result_noisy_01 = result_noisy

        return result_noisy_01
