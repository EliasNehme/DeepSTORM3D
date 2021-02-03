# Import modules and libraries
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
from DeepSTORM3D.physics_utils import EmittersToPhases
from DeepSTORM3D.helper_utils import normalize_01
from skimage.io import imread


# ======================================================================================================================
# numpy array conversion to variable and numpy complex conversion to 2 channel torch tensor
# ======================================================================================================================


# function converts numpy array on CPU to torch Variable on GPU
def to_var(x):
    """
    Input is a numpy array and output is a torch variable with the data tensor
    on cuda.
    """

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# function converts numpy array on CPU to torch Variable on GPU
def complex_to_tensor(phases_np):
    Nbatch, Nemitters, Hmask, Wmask = phases_np.shape
    phases_torch = torch.zeros((Nbatch, Nemitters, Hmask, Wmask, 2)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 0] = torch.from_numpy(np.real(phases_np)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 1] = torch.from_numpy(np.imag(phases_np)).type(torch.FloatTensor)
    return phases_torch


# ======================================================================================================================
# continuous emitter positions sampling using two steps: first sampling disjoint indices on a coarse 3D grid,
# and afterwards refining each index using a local perturbation.
# ======================================================================================================================


# Define a batch data generator for training and testing
def generate_batch(batch_size, setup_params, seed=None):

    # if we're testing then seed the random generator
    if seed is not None:
        np.random.seed(seed)
    
    # randomly vary the number of emitters
    num_particles_range = setup_params['num_particles_range']
    num_particles = np.asscalar(np.random.randint(num_particles_range[0], num_particles_range[1], 1))
    
    # distrbution for sampling the number of counts per emitter
    if setup_params['nsig_unif']:

        # uniformly distributed counts per emitter (counts == photons assuming gain=1)
        Nsig_range = setup_params['nsig_unif_range']
        Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles))
        Nphotons = Nphotons.astype('float32')
    else:

        # gamma distributed counts per emitter
        gamma_params = setup_params['nsig_gamma_params']
        Nphotons = np.random.gamma(gamma_params[0], gamma_params[1], (batch_size, num_particles))
        Nphotons = Nphotons.astype('float32')

    # calculate upsampling factor
    pixel_size_FOV, pixel_size_rec = setup_params['pixel_size_FOV'], setup_params['pixel_size_rec']
    upsampling_factor = pixel_size_FOV / pixel_size_rec

    # update dimensions
    H, W, D, clear_dist = setup_params['H'], setup_params['W'], setup_params['D'], setup_params['clear_dist']
    H, W, clear_dist = int(H * upsampling_factor), int(W * upsampling_factor), int(clear_dist * upsampling_factor)

    # for each sample in the batch generate unique grid positions
    xyz_grid = np.zeros((batch_size, num_particles, 3)).astype('int')
    for k in range(batch_size):
        
        # randomly choose num_particles linear indices
        lin_ind = np.random.randint(0, (W - clear_dist * 2) * (H - clear_dist * 2) * D,
                                    num_particles)

        # switch from linear indices to subscripts
        zgrid_vec, ygrid_vec, xgrid_vec = np.unravel_index(lin_ind, (
        D, H - 2 * clear_dist, W - 2 * clear_dist))

        # reshape subscripts to fit into the 3D on grid array
        xyz_grid[k, :, 0] = np.reshape(xgrid_vec, (1, num_particles), 'F') + clear_dist
        xyz_grid[k, :, 1] = np.reshape(ygrid_vec, (1, num_particles), 'F') + clear_dist
        xyz_grid[k, :, 2] = np.reshape(zgrid_vec, (1, num_particles), 'F')

    # for each grid position add a continuous shift inside the voxel within [-0.5,0.5)
    x_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    y_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    z_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    
    # minimal height in z and axial pixel size
    zmin, pixel_size_axial = setup_params['zmin'], setup_params['pixel_size_axial']

    # group samples into an array of size [batch_size, emitters, xyz]
    xyz = np.zeros((batch_size, num_particles, 3))
    xyz[:, :, 0] = (xyz_grid[:, :, 0] - int(np.floor(W / 2)) + x_local + 0.5) * pixel_size_rec
    xyz[:, :, 1] = (xyz_grid[:, :, 1] - int(np.floor(H / 2)) + y_local + 0.5) * pixel_size_rec
    xyz[:, :, 2] = (xyz_grid[:, :, 2] + z_local + 0.5) * pixel_size_axial + zmin

    # resulting batch of data
    return xyz, Nphotons


# ======================================================================================================================
# projection of the continuous positions on the recovery grid in order to generate the training label
# ======================================================================================================================


# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, setup_params):
    
    # calculate upsampling factor
    pixel_size_FOV, pixel_size_rec = setup_params['pixel_size_FOV'], setup_params['pixel_size_rec']
    upsampling_factor = pixel_size_FOV / pixel_size_rec
    
    # axial pixel size
    pixel_size_axial = setup_params['pixel_size_axial']

    # current dimensions
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']
    
    # shift the z axis back to 0
    zshift = xyz_np[:, :, 2] - setup_params['zmin']
    
    # number of particles
    batch_size, num_particles = zshift.shape
    
    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor(xyz_np[:, :, 0]/pixel_size_rec) + np.floor(W/2)*upsampling_factor).astype('int')
    yg = (np.floor(xyz_np[:, :, 1]/pixel_size_rec) + np.floor(H/2)*upsampling_factor).astype('int')
    zg = (np.floor(zshift/pixel_size_axial)).astype('int')
    
    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

    # update dimensions
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    
    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indY, indX])
    else:
        ibool = torch.LongTensor([indZ, indY, indX])
    
    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    
    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    else:
        boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([D, H, W])).to_dense()
    
    return boolean_grid


# ======================================================================================================================
# dataset class instantiation for both pre-calculated images / training positions to accelerate data loading in training
# ======================================================================================================================


# PSF images with corresponding xyz labels dataset
class ImagesDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
        self.train_stats = setup_params['train_stats']
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    # sampling one example from the data
    def __getitem__(self, index):
        
        # select sample
        ID = self.list_IDs[index]
        
        # load tiff image
        im_name = self.root_dir + '/im' + ID + '.tiff'
        im_np = imread(im_name)

        # turn image into torch tensor with 1 channel
        im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)
        
        return im_tensor, bool_grid


# xyz and photons turned online to fourier phases dataset
class PhasesOnlineDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, list_IDs, labels, setup_params):
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    # sampling one example from the data
    def __getitem__(self, index):
        
        # select sample
        ID = self.list_IDs[index]
        
        # associated number of photons
        dict = self.labels[ID]
        Nphotons_np = dict['N']
        Nphotons = torch.from_numpy(Nphotons_np)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = dict['xyz']        
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)
        
        # calculate phases and turn them to tensors
        phases_np = EmittersToPhases(xyz_np, self.setup_params)
        phases_tensor = complex_to_tensor(phases_np)

        return phases_tensor, Nphotons, bool_grid


# Experimental images with normalization dataset
class ExpDataset(Dataset):

    # initialization of the dataset
    def __init__(self, im_list, setup_params):
        self.im_list = im_list
        self.project_01 = setup_params['project_01']
        if self.project_01 is False:
            self.global_factors = setup_params['global_factors']
        self.train_stats = setup_params['train_stats']

    # total number of samples in the dataset
    def __len__(self):
        return len(self.im_list)

    # sampling one example from the data
    def __getitem__(self, index):

        # load tif image
        im_np = imread(self.im_list[index])
        im_np = im_np.astype("float32")

        # normalize image according to the training setting
        if self.project_01 is True:
            im_np = normalize_01(im_np)
        else:
            im_np = (im_np - self.global_factors[0]) / self.global_factors[1]

        # alter the mean and std to match the training set
        if self.project_01 is True:
            im_np = (im_np - im_np.mean()) / im_np.std()
            im_np = im_np * self.train_stats[1] + self.train_stats[0]

        # turn image into torch tensor with 1 channel
        im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np)

        return im_tensor

# ======================================================================================================================
# Sorting function in case glob uploads images in the wrong order
# ======================================================================================================================

# ordering file names according to number
def sort_names_tif(img_names):
    nums = []
    for i in img_names:
        i2 = i .split(".tif")
        i3 = i2[0].split("/")
        nums.append(int(i3[-1]))
    indices = np.argsort(np.array(nums))
    fixed_names = [img_names[i] for i in indices]
    return fixed_names


