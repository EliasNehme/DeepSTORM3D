# Import modules and libraries
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor
from DeepSTORM3D.physics_utils import calc_bfp_grids, EmittersToPhases, PhysicalLayer
from DeepSTORM3D.helper_utils import CalcMeanStd_All
import os
import pickle
from PIL import Image
import argparse


# generate training data (either images for localization cnn or locations and intensities for psf learning)
def gen_data(setup_params):

    # random seed for repeatability
    torch.manual_seed(999)
    np.random.seed(566)

    # calculate on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True

    # calculate the effective sensor pixel size taking into account magnification and set the recovery pixel size to be
    # the same such that sampling of training positions is performed on this coarse grid
    setup_params['pixel_size_FOV'] = setup_params['pixel_size_CCD'] / setup_params['M']  # in [um]
    setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 1  # in [um]

    # calculate the axial range and the axial pixel size depending on the volume discretization
    setup_params['axial_range'] = setup_params['zmax'] - setup_params['zmin']  # [um]
    setup_params['pixel_size_axial'] = setup_params['axial_range'] / setup_params['D']  # [um]

    # calculate back focal plane grids and needed terms for on the fly PSF calculation
    setup_params = calc_bfp_grids(setup_params)

    # training data folder for saving
    path_train = setup_params['training_data_path']
    if not (os.path.isdir(path_train)):
        os.mkdir(path_train)

    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)

    # batch size for generating training examples:
    # locations for phase mask learning are saved in batches of 16 for convenience
    if setup_params['learn_mask']:
        batch_size_gen = 16
    else:
        batch_size_gen = 1
    setup_params['batch_size_gen'] = batch_size_gen

    # calculate the number of training batches to sample
    ntrain_batches = int(setup_params['ntrain'] / batch_size_gen)
    setup_params['ntrain_batches'] = ntrain_batches

    # phase mask and psf module for simulation
    if setup_params['learn_mask'] is False:
        mask_init = setup_params['mask_init']
        if mask_init is None:
            print('If the training mode is not set to learning a phase mask then you should supply a phase mask')
            return
        else:
            mask_param = torch.from_numpy(mask_init)
            psf_module = PhysicalLayer(setup_params)

    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):

        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen, setup_params)

        # if we intend to learn a mask, simply save locations and intensities
        if setup_params['learn_mask']:

            # save xyz, N, to labels dict
            labels_dict[str(i)] = {'xyz': xyz, 'N': Nphotons}

        # otherwise, create the respective image and save only the locations
        else:

            # calculate phases from simulated locations
            phase_emitters = EmittersToPhases(xyz, setup_params)

            # cast phases and number of photons to tensors
            Nphotons_tensor = torch.from_numpy(Nphotons).type(torch.FloatTensor)
            phases_tensor = complex_to_tensor(phase_emitters)

            # pass them through the physical layer to get the corresponding image
            im = psf_module(mask_param.to(device), phases_tensor.to(device), Nphotons_tensor.to(device))
            im_np = np.squeeze(im.data.cpu().numpy())

            # normalize image according to the global factors assuming it was not projected to [0,1]
            if setup_params['project_01'] is False:
                im_np = (im_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # look at the image if specified
            if setup_params['visualize']:

                # squeeze batch dimension in xyz
                xyz2 = np.squeeze(xyz, 0)

                # plot the image and the simulated xy centers on top
                fig1 = plt.figure(1)
                imfig = plt.imshow(im_np, cmap='gray')
                pixel_size_FOV, W, H = setup_params['pixel_size_FOV'], setup_params['W'], setup_params['H']
                plt.plot(xyz2[:, 0] / pixel_size_FOV + np.floor(W / 2), xyz2[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
                plt.title(str(i))
                fig1.colorbar(imfig)
                plt.draw()
                plt.pause(0.05)
                plt.clf()

            # threshold out dim emitters if counts are gamma distributed
            if (setup_params['nsig_unif'] is False) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > setup_params['nsig_thresh'], :]

            # save image as a tiff file and xyz to labels dict
            im_name_tiff = path_train + 'im' + str(i) + '.tiff'
            img1 = Image.fromarray(im_np)
            img1.save(im_name_tiff)
            labels_dict[str(i)] = xyz

        # print number of example
        print('Training Example [%d / %d]' % (i + 1, ntrain_batches))

    # calculate training set mean and standard deviation (if we are generating images)
    if setup_params['learn_mask'] is False:
        train_stats = CalcMeanStd_All(path_train, labels_dict)
        setup_params['train_stats'] = train_stats

    # print status
    print('=' * 50)
    print('Sampling examples for validation')
    print('=' * 50)

    # calculate the number of training batches to sample
    nvalid_batches = int(setup_params['nvalid'] // batch_size_gen)
    setup_params['nvalid_batches'] = nvalid_batches

    # set the number of particles to the middle of the range for validation
    num_particles_range = setup_params['num_particles_range']
    setup_params['num_particles_range'] = [num_particles_range[1]//2, num_particles_range[1]//2 + 1]

    # sample validation examples
    for i in range(nvalid_batches):

        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen, setup_params)

        # if we intend to learn a mask, simply save locations and intensities
        if setup_params['learn_mask']:

            # save xyz, N, to labels dict
            labels_dict[str(i + ntrain_batches)] = {'xyz': xyz, 'N': Nphotons}

        # otherwise, create the respective image and save only the locations
        else:

            # calculate phases from simulated locations
            phase_emitters = EmittersToPhases(xyz, setup_params)

            # cast phases and number of photons to tensors
            Nphotons_tensor = torch.from_numpy(Nphotons).type(torch.FloatTensor)
            phases_tensor = complex_to_tensor(phase_emitters)

            # pass them through the physical layer to get the corresponding image
            im = psf_module(mask_param.to(device), phases_tensor.to(device), Nphotons_tensor.to(device))
            im_np = np.squeeze(im.data.cpu().numpy())

            # normalize image according to the global factors assuming it was not projected to [0,1]
            if setup_params['project_01'] is False:
                im_np = (im_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # look at the image if specified
            if setup_params['visualize']:

                # squeeze batch dimension in xyz
                xyz2 = np.squeeze(xyz, 0)

                # plot the image and the simulated xy centers on top
                fig1 = plt.figure(1)
                imfig = plt.imshow(im_np, cmap='gray')
                pixel_size_FOV, W, H = setup_params['pixel_size_FOV'], setup_params['W'], setup_params['H']
                plt.plot(xyz2[:, 0] / pixel_size_FOV + np.floor(W / 2), xyz2[:, 1] / pixel_size_FOV + np.floor(H / 2),
                         'r+')
                plt.title(str(i))
                fig1.colorbar(imfig)
                plt.draw()
                plt.pause(0.05)
                plt.clf()

            # threshold out dim emitters if counts are gamma distributed
            if (setup_params['nsig_unif'] is False) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > setup_params['nsig_thresh'], :]

            # save image as a tiff file and xyz to labels dict
            im_name_tiff = path_train + 'im' + str(i + ntrain_batches) + '.tiff'
            img1 = Image.fromarray(im_np)
            img1.save(im_name_tiff)
            labels_dict[str(i + ntrain_batches)] = xyz

        # print number of example
        print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))

    # save all xyz's dictionary as a pickle file
    path_labels = path_train + 'labels.pickle'
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # set the number of particles back to the specified range
    setup_params['num_particles_range'] = num_particles_range

    # partition built in simulation
    ind_all = np.arange(0, ntrain_batches + nvalid_batches, 1)
    list_all = ind_all.tolist()
    list_IDs = [str(i) for i in list_all]
    train_IDs = list_IDs[:ntrain_batches]
    valid_IDs = list_IDs[ntrain_batches:]
    partition = {'train': train_IDs, 'valid': valid_IDs}
    setup_params['partition'] = partition

    # update recovery pixel in xy to be x4 smaller if we are learning a localization net
    if setup_params['learn_mask'] is False:
        setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 4  # in [um]

    # save setup parameters dictionary for training and testing
    path_setup_params = path_train + 'setup_params.pickle'
    with open(path_setup_params, 'wb') as handle:
        pickle.dump(setup_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print status
    print('Finished sampling examples!')

    # close figure if it was open for visualization
    if (setup_params['learn_mask'] is False) and setup_params['visualize']:
        plt.close(fig1)


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously wrapped settings dictionary
    parser.add_argument('--setup_params', help='path to the parameters wrapped in the script parameter_setting', required=True)

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    gen_data(args.setup_params)
