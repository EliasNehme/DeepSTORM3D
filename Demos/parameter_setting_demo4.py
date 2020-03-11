# this script encapsulates all needed parameters for training/learning a phase mask

# import needed libraries
from math import pi
import os


def demo4_parameters():

    # path to current directory
    path_curr_dir = os.getcwd()

    # ======================================================================================
    # initial mask and training mode
    # ======================================================================================

    # boolean that specifies whether we are learning a mask or not
    learn_mask = True

    # initial mask for learning an optimized mask or final mask for training a localization model
    # if learn_mask=True the initial mask is initialized by default to be zero-modulation
    mask_init = None

    # mask options dictionary
    mask_opts = {'learn_mask': learn_mask, 'mask_init': mask_init}

    # ======================================================================================
    # optics settings: objective, light, sensor properties
    # ======================================================================================

    lamda = 0.58  # mean emission wavelength # in [um] (1e-6*meter)
    NA = 1.45  # numerical aperture of the objective lens
    noil = 1.518  # immersion medium refractive index
    nwater = 1.33  # imaging medium refractive index
    pixel_size_CCD = 11  # sensor pixel size in [um] (including binning)
    pixel_size_SLM = 24  # SLM pixel size in [um] (after binning of 3 to reduce computational complexity)
    M = 100  # optical magnification
    f_4f = 15e4  # 4f lenses focal length in [um]

    # optical settings dictionary
    optics_dict = {'lamda': lamda, 'NA': NA, 'noil': noil, 'nwater': nwater, 'pixel_size_CCD': pixel_size_CCD,
                   'pixel_size_SLM': pixel_size_SLM, 'M': M, 'f_4f': f_4f}

    # ======================================================================================
    # phase mask and image space dimensions for simulation
    # ======================================================================================

    # phase mask dimensions
    Hmask, Wmask = 329, 329  # in SLM [pixels]

    # single training image dimensions
    H, W = 121, 121  # in sensor [pixels]

    # safety margin from the boundary to prevent PSF truncation
    clear_dist = 20  # in sensor [pixels]

    # training z-range anf focus
    zmin = 0  # minimal z in [um] (including the axial shift)
    zmax = 5  # maximal z in [um] (including the axial shift)
    NFP = 2.5  # nominal focal plane in [um] (including the axial shift)

    # discretization in z
    D = 51  # in [voxels] spanning the axial range (zmax - zmin)

    # data dimensions dictionary
    data_dims_dict = {'Hmask': Hmask, 'Wmask': Wmask, 'H': H, 'W': W, 'clear_dist': clear_dist, 'zmin': zmin,
                      'zmax': zmax, 'NFP': NFP, 'D': D}

    # ======================================================================================
    # number of emitters in each FOV
    # ======================================================================================

    # upper and lower limits for the number fo emitters
    num_particles_range = [1, 35]

    # number of particles dictionary
    num_particles_dict = {'num_particles_range': num_particles_range}

    # ======================================================================================
    # signal counts distribution and settings
    # ======================================================================================

    # boolean that specifies whether the signal counts are uniformly distributed
    nsig_unif = True

    # range of signal counts assuming a uniform distribution
    nsig_unif_range = [10000, 60000]  # in [counts]

    # parameters for sampling signal counts assuming a gamma distribution
    nsig_gamma_params = None  # in [counts]

    # threshold on signal counts to discard positions from the training labels
    nsig_thresh = None  # in [counts]

    # signal counts dictionary
    nsig_dict = {'nsig_unif': nsig_unif, 'nsig_unif_range': nsig_unif_range, 'nsig_gamma_params': nsig_gamma_params,
                 'nsig_thresh': nsig_thresh}

    # ======================================================================================
    # blur standard deviation for smoothing PSFs to match experimental conditions
    # ======================================================================================

    # upper and lower blur standard deviation for each emitter to account for finite size
    blur_std_range = [0.75, 1.25]  # in sensor [pixels]

    # blur dictionary
    blur_dict = {'blur_std_range': blur_std_range}

    # ======================================================================================
    # uniform/non-uniform background settings
    # ======================================================================================

    # uniform background value per pixel
    unif_bg = 0  # in [counts]

    # boolean flag whether or not to include a non-uniform background
    nonunif_bg_flag = True

    # maximal offset for the center of the non-uniform background in pixels
    nonunif_bg_offset = [10, 10]  # in sensor [pixels]

    # peak and valley minimal values for the super-gaussian; randomized with addition of up to 50%
    nonunif_bg_minvals = [20.0, 100.0]  # in [counts]

    # minimal and maximal angle of the super-gaussian for augmentation
    nonunif_bg_theta_range = [-pi/4, pi/4]  # in [radians]

    # nonuniform background dictionary
    nonunif_bg_dict = {'nonunif_bg_flag': nonunif_bg_flag, 'unif_bg': unif_bg, 'nonunif_bg_offset': nonunif_bg_offset,
                       'nonunif_bg_minvals': nonunif_bg_minvals, 'nonunif_bg_theta_range': nonunif_bg_theta_range}

    # ======================================================================================
    # read noise settings
    # ======================================================================================

    # boolean flag whether or not to include read noise
    read_noise_flag = False

    # flag whether of not the read noise standard deviation is not uniform across the FOV
    read_noise_nonuinf = None

    # range of baseline of the min-subtracted data in STORM
    read_noise_baseline_range = None  # in [counts]

    # read noise standard deviation upper and lower range
    read_noise_std_range = None  # in [counts]

    # read noise dictionary
    read_noise_dict = {'read_noise_flag': read_noise_flag, 'read_noise_nonuinf': read_noise_nonuinf,
                       'read_noise_baseline_range': read_noise_baseline_range,
                       'read_noise_std_range': read_noise_std_range}

    # ======================================================================================
    # image normalization settings
    # ======================================================================================

    # boolean flag whether or not to project the images to the range [0, 1]
    project_01 = True

    # global normalization factors for STORM (subtract the first and divide by the second)
    global_factors = None  # in [counts]

    # image normalization dictionary
    norm_dict = {'project_01': project_01, 'global_factors': global_factors}

    # ======================================================================================
    # training data settings
    # ======================================================================================

    # number of training and validation examples
    ntrain = 9000
    nvalid = 1000

    # path for saving training examples: images + locations for localization net or locations + photons for PSF learning
    training_data_path = path_curr_dir + "/TrainingLocations_demo4/"

    # boolean flag whether to visualize examples while created
    visualize = True

    # training data dictionary
    training_dict = {'ntrain': ntrain, 'nvalid': nvalid, 'training_data_path': training_data_path, 'visualize': visualize}

    # ======================================================================================
    # learning settings
    # ======================================================================================

    # results folder to save the trained model
    results_path = path_curr_dir + "/Results_demo4/"

    # maximal dilation flag when learning a localization CNN (set to None if learn_mask=True as we use a different CNN)
    dilation_flag = True  # if set to 1 then dmax=16 otherwise dmax=4

    # batch size for training a localization model (set to 1 for mask learning as examples are generated 16 at a time)
    batch_size = 1

    # maximal number of epochs
    max_epochs = 50

    # initial learning rate for adam
    initial_learning_rate = 0.0005

    # scaling factor for the loss function
    scaling_factor = 100.0

    # learning dictionary
    learning_dict = {'results_path': results_path, 'dilation_flag': dilation_flag, 'batch_size': batch_size,
                     'max_epochs': max_epochs, 'initial_learning_rate': initial_learning_rate,
                     'scaling_factor': scaling_factor}

    # ======================================================================================
    # resuming from checkpoint settings
    # ======================================================================================

    # boolean flag whether to resume training from checkpoint
    resume_training = False

    # number of epochs to resume training
    num_epochs_resume = None

    # saved checkpoint to resume from
    checkpoint_path = None

    # checkpoint dictionary
    checkpoint_dict = {'resume_training': resume_training, 'num_epochs_resume': num_epochs_resume,
                       'checkpoint_path': checkpoint_path}

    # ======================================================================================
    # final resulting dictionary including all parameters
    # ======================================================================================

    settings = {**mask_opts, **num_particles_dict, **nsig_dict, **blur_dict, **nonunif_bg_dict, **read_noise_dict,
                **norm_dict, **optics_dict, **data_dims_dict, **training_dict, **learning_dict, **checkpoint_dict}

    return settings


if __name__ == '__main__':
    parameters = demo4_parameters()

