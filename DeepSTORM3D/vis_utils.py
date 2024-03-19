# Import modules and libraries
import torch.nn as nn
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift
from DeepSTORM3D.physics_utils import Croplayer, Normalize01, MaskPhasesToPSFs, BlurLayer, NoiseLayer


# show current PSF given the phase mask and the defocus term
def ShowMaskPSF(phasemask, vis_term, zvis, iternum=None):

    # extract phase mask from the model and convert it to numpy
    phasemask_numpy = phasemask.data.cpu().clone().numpy()

    # phase term due to the phase mask
    phasemask_term = np.exp(1j * phasemask_numpy)

    # total phase term
    phase3D = phasemask_term * vis_term

    # calculate the centered fourier transform on the H, W dims
    fft_res = fftshift(fft2(ifftshift(phase3D, axes=(1, 2)), norm="ortho"), axes=(1, 2))

    # take the absolute value squared
    PSFs = np.abs(fft_res) ** 2

    # sizes of the SLM mask to crop middle portions
    H, W = phasemask_numpy.shape
    ch, cw = H // 2, W // 2
    delta_mask, delta_psf = 90, 30

    # cropped phase mask middle
    phasemask_center = phasemask_numpy[ch - delta_mask - 1:ch + delta_mask, cw - delta_mask - 1:cw + delta_mask]

    # smoothed and cropped PSFs
    PSFs_final = np.zeros((5, 61, 61))
    for i in range(5):
        PSFs_final[i, :, :] = gaussian_filter(PSFs[i, ch - delta_psf - 1:ch + delta_psf,
                                              cw - delta_psf - 1:cw + delta_psf], sigma=(1, 1), mode='constant')

    # plot 5 PSFs along the axis
    plt.clf()
    plt.subplot(2, 5, 3)
    im = plt.imshow(phasemask_center)
    plt.title("phase mask")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    for i in range(5):
        plt.subplot(2, 5, 5 + i + 1)
        im = plt.imshow(PSFs_final[i, :, :])
        if i < 5:
            plt.title('z = ' + str(zvis[i]) + ' um')
        else:
            plt.title('out of range')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis("off")
    if iternum is not None:
        plt.suptitle("Iteration # %d" % (iternum + 1))
    plt.draw()
    plt.pause(0.05)


# show training and validation loss at the end of each epoch
def ShowLossJaccardAtEndOfEpoch(learning_results, epoch):

    # x axis for the plot
    steps_per_epoch = learning_results['steps_per_epoch']
    iter_axis = np.arange(steps_per_epoch, steps_per_epoch * (epoch + 1) + 1, steps_per_epoch)        

    # plot result
    plt.clf()
    plt.subplot(4, 1, 1)
    linet, linev = plt.plot(iter_axis, learning_results['train_loss'], '-og', iter_axis, learning_results['valid_loss'], '-^r')
    plt.ylabel("Mean Loss")
    plt.legend((linet, linev), ('Train', 'Valid'))
    plt.title("Training Metrics at Epoch %d" % (epoch + 1))
    plt.subplot(4, 1, 2)
    plt.plot(iter_axis, learning_results['train_jacc'], '-og', iter_axis, learning_results['valid_jacc'], '-^r')
    plt.ylabel("Jaccard Index")
    plt.subplot(4, 1, 3)
    plt.plot(iter_axis, learning_results['sum_valid'], 'r')
    plt.ylabel("Mean Sum of Validation")
    plt.subplot(4, 1, 4)
    plt.plot(iter_axis, learning_results['max_valid'], 'r')
    plt.ylabel("Maximum of Validation")
    plt.draw()
    plt.pause(0.05)


# show the recovered 3D positions alongside the ground truth ones
def ShowRecovery3D(xyz_gt, xyz_rec):

    # define a figure for 3D scatter plot
    ax = plt.axes(projection='3d')
    
    # plot boolean recoveries in 3D
    ax.scatter(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], c='b', marker='o', label='GT', depthshade=False)
    ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='Rec', depthshade=False)

    # add labels and and legend
    ax.set_xlabel('X [um]')
    ax.set_ylabel('Y [um]')
    ax.set_zlabel('Z [um]')
    plt.legend()


# physical layer taking 3D continuous coordinates and a mask and outputting 2D images for visualization
class PhysicalLayerVisualization(nn.Module):
    def __init__(self, setup_params, blur_flag, noise_flag, norm_flag):
        super(PhysicalLayerVisualization, self).__init__()
        self.device = setup_params['device']
        self.mask = MaskPhasesToPSFs(setup_params)
        if blur_flag:
            std_max = setup_params['blur_std_range'][1]
            setup_params['blur_std_range'] = [std_max, std_max]
        else:
            std_min = setup_params['blur_std_range'][0]
            setup_params['blur_std_range'] = [std_min, std_min]
        self.blur = BlurLayer(setup_params)
        self.crop = Croplayer(setup_params)
        self.noise = NoiseLayer(setup_params)
        self.norm01 = Normalize01()
        self.noise_flag = noise_flag
        self.norm_flag = norm_flag

    def forward(self, mask, phase_emitter, nphotons):
        
        # generate the PSF images from the phase mask, xyz locations, and orientation params
        PSF4D = self.mask(mask, phase_emitter, nphotons)
        
        # crop relevant FOV area
        images4D_crop = self.crop(PSF4D)
        
        # blur each emitter with slightly different gaussian
        images4D_crop_blur = self.blur(images4D_crop)

        # apply the measurement noise model
        if self.noise_flag:
            result_noisy = self.noise(images4D_crop_blur)
        else:
            result_noisy = images4D_crop_blur

        # [0,1] normalization to prevent scaling volnurability
        if self.norm_flag:
            result_noisy_01 = self.norm01(result_noisy)
        else:
            result_noisy_01 = result_noisy

        return result_noisy_01


# show PSF image
def ShowRecNetInput(input_var, title_str):

    # extract phase mask from the model and convert it to numpy
    net_input = np.squeeze(input_var.data.cpu().numpy())

    # plot it as func. of iterations
    plt.figure()
    plt.imshow(net_input)
    plt.colorbar()
    plt.title(title_str)

