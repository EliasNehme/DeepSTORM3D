# Import modules and libraries
import numpy as np
import glob
import time
from datetime import datetime
import argparse
import csv
import pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor, ExpDataset, sort_names_tif
from DeepSTORM3D.cnn_utils import LocalizationCNN
from DeepSTORM3D.vis_utils import ShowMaskPSF, ShowRecovery3D, ShowLossJaccardAtEndOfEpoch
from DeepSTORM3D.vis_utils import PhysicalLayerVisualization, ShowRecNetInput
from DeepSTORM3D.physics_utils import EmittersToPhases, calc_bfp_grids
from DeepSTORM3D.postprocess_utils import Postprocess
from DeepSTORM3D.assessment_utils import calc_jaccard_rmse
from DeepSTORM3D.helper_utils import normalize_01, xyz_to_nm


def test_model(path_results, postprocess_params, exp_imgs_path=None, seed=66):

    # close all existing plots
    plt.close("all")

    # load assumed setup parameters
    path_params_pickle = path_results + 'setup_params.pickle'
    with open(path_params_pickle, 'rb') as handle:
        setup_params = pickle.load(handle)

    # run on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True

    # phase term for PSF visualization
    vis_term, zvis = setup_params['vis_term'], setup_params['zvis']

    # phase mask for visualization
    mask_param = torch.from_numpy(setup_params['mask_init']).to(device)

    # plot used mask and PSF
    plt.figure(figsize=(10,5))
    ShowMaskPSF(mask_param, vis_term, zvis)

    # load learning results
    path_learning_pickle = path_results + 'learning_results.pickle'
    with open(path_learning_pickle, 'rb') as handle:
        learning_results = pickle.load(handle)

    # plot metrics evolution in training for debugging
    plt.figure()
    ShowLossJaccardAtEndOfEpoch(learning_results, learning_results['epoch_converged'])

    # build model and convert all the weight tensors to GPU is available
    cnn = LocalizationCNN(setup_params)
    cnn.to(device)

    # load learned weights
    cnn.load_state_dict(torch.load(path_results + 'weights_best_loss.pkl'))

    # post-processing module on CPU/GPU
    thresh, radius, keep_singlez = postprocess_params['thresh'], postprocess_params['radius'], postprocess_params['keep_singlez']
    postprocessing_module = Postprocess(setup_params, thresh, radius, keep_singlez)

    # if no experimental imgs are supplied then sample a random example
    if exp_imgs_path is None:

        # visualization module to visualize the 3D positions recovered by the net as images
        psf_module_vis = PhysicalLayerVisualization(setup_params, 0, 0, 1)

        # ==============================================================================================================
        # generate a simulated test image
        # ==============================================================================================================

        # set random number generators given the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # sample a single piece of data
        xyz_gt, nphotons_gt = generate_batch(1, setup_params)

        # calculate phases and cast them to device
        phases_np = EmittersToPhases(xyz_gt, setup_params)
        phases_emitter_gt = complex_to_tensor(phases_np).to(device)

        # initialize the physical layer that encodes xyz into noisy PSFs
        psf_module_net = PhysicalLayerVisualization(setup_params, 1, 1, 0)

        # pass xyz and N through the physical layer to get the simulated image
        nphotons_gt = torch.from_numpy(nphotons_gt).type(torch.FloatTensor).to(device)
        test_input_im = psf_module_net(mask_param, phases_emitter_gt, nphotons_gt)

        # normalize image according to the training setting
        if setup_params['project_01'] is True:
            test_input_im = normalize_01(test_input_im)
        else:
            test_input_im = (test_input_im - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

        # alter the mean and std to match the training set
        """
        if setup_params['project_01'] is True:
            test_input_im = (test_input_im - test_input_im.mean())/test_input_im.std()
            test_input_im = test_input_im*setup_params['train_stats'][1] + setup_params['train_stats'][0]
        """

        # ==============================================================================================================
        # predict the positions by post-processing the net's output
        # ==============================================================================================================

        # prediction using model
        cnn.eval()
        with torch.set_grad_enabled(False):
            pred_volume = cnn(test_input_im)

        # post-process predicted volume
        tpost_start = time.time()
        xyz_rec, conf_rec = postprocessing_module(pred_volume)
        tpost_elapsed = time.time() - tpost_start
        print('Post-processing complete in {:.6f}s'.format(tpost_elapsed))

        # time prediction using model after first forward pass which is slow
        cnn.eval()
        tinf_start = time.time()
        with torch.set_grad_enabled(False):
            pred_volume = cnn(test_input_im)
        tinf_elapsed = time.time() - tinf_start
        print('Inference complete in {:.6f}s'.format(tinf_elapsed))

        # take out dim emitters from GT
        if setup_params['nsig_unif'] is False:
            nemitters = xyz_gt.shape[1]
            if np.not_equal(nemitters, 1):
                nphotons_gt = np.squeeze(nphotons_gt, 0)
                xyz_gt = xyz_gt[:, nphotons_gt > setup_params['nsig_thresh'], :]

        # plot recovered 3D positions compared to GT
        plt.figure()
        xyz_gt = np.squeeze(xyz_gt, 0)
        ShowRecovery3D(xyz_gt, xyz_rec)

        # report the number of found emitters
        print('Found {:d} emitters out of {:d}'.format(xyz_rec.shape[0], xyz_gt.shape[0]))

        # calculate quantitative metrics assuming a matching radius of 100 nm
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)

        # report quantitative metrics
        print('Jaccard Index = {:.2f}%, Lateral RMSE = {:.2f} nm, Axial RMSE = {:.2f}'.format(
            jaccard_index*100, RMSE_xy*1e3, RMSE_z*1e3))

        # ==============================================================================================================
        # compare the network positions to the input image
        # ==============================================================================================================

        # turn recovered positions into phases
        xyz_rec = np.expand_dims(xyz_rec, 0)
        phases_np = EmittersToPhases(xyz_rec, setup_params)
        phases_emitter_rec = complex_to_tensor(phases_np).to(device)

        # use a uniform number of photons for recovery visualization
        nphotons_rec = 5000 * torch.ones((1, xyz_rec.shape[1])).to(device)

        # generate the recovered image by the net
        test_pred_im = psf_module_vis(mask_param, phases_emitter_rec, nphotons_rec)

        # compare the recovered image to the input
        ShowRecNetInput(test_input_im, 'Simulated Input to Localization Net')
        ShowRecNetInput(test_pred_im, 'Recovered Input Matching Net Localizations')

        # return recovered locations and net confidence
        return np.squeeze(xyz_rec, 0), conf_rec

    else:

        # read all imgs in the experimental data directory assuming ".tif" extension
        img_names = glob.glob(exp_imgs_path + '*.tif')

        # if given only 1 image then show xyz in 3D and recovered image
        if len(img_names) == 1:

            # ==========================================================================================================
            # read experimental image and normalize it
            # ==========================================================================================================

            # read exp image in  uint16
            exp_im = imread(img_names[0])
            exp_img = exp_im.astype("float32")

            # normalize image according to the training setting
            if setup_params['project_01'] is True:
                exp_img = normalize_01(exp_img)
            else:
                exp_img = (exp_img - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # alter the mean and std to match the training set
            """
            if setup_params['project_01'] is True:
                exp_img = (exp_img - exp_img.mean()) / exp_img.std()
                exp_img = exp_img * setup_params['train_stats'][1] + setup_params['train_stats'][0]
            """
            
            # turn image into torch tensor with 1 channel on GPU
            exp_img = np.expand_dims(exp_img, 0)
            exp_img = np.expand_dims(exp_img, 0)
            exp_tensor = torch.FloatTensor(exp_img).to(device)

            # ==========================================================================================================
            # predict the positions by post-processing the net's output
            # ==========================================================================================================

            # prediction using model
            cnn.eval()
            with torch.set_grad_enabled(False):
                pred_volume = cnn(exp_tensor)

            # post-process predicted volume
            tpost_start = time.time()
            xyz_rec, conf_rec = postprocessing_module(pred_volume)
            tpost_elapsed = time.time() - tpost_start
            print('Post-processing complete in {:.6f}s'.format(tpost_elapsed))

            # time prediction using model after first forward pass which is slow
            cnn.eval()
            tinf_start = time.time()
            with torch.set_grad_enabled(False):
                pred_volume = cnn(exp_tensor)
            tinf_elapsed = time.time() - tinf_start
            print('Inference complete in {:.6f}s'.format(tinf_elapsed))

            # plot recovered 3D positions compared to GT
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='DL', depthshade=False)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_zlabel('Z [um]')
            plt.gca().invert_yaxis()
            plt.title('3D Recovered Positions')

            # report the number of found emitters
            print('Found {:d} emitters'.format(xyz_rec.shape[0]))

            # ==========================================================================================================
            # compare the network positions to the input image
            # ==========================================================================================================

            # visualization module to visualize the 3D positions recovered by the net as images
            H, W = exp_im.shape
            setup_params['H'], setup_params['W'] = H, W
            psf_module_vis = PhysicalLayerVisualization(setup_params, 0, 0, 1)
            
            # if the grid is too large then pre-compute the phase grids again
            if H > setup_params['Hmask'] or W > setup_params['Wmask']:
                setup_params['pixel_size_SLM'] /= 2
                setup_params['Hmask'] *= 2
                setup_params['Wmask'] *= 2
                setup_params = calc_bfp_grids(setup_params)
                mask_param = interpolate(mask_param.unsqueeze(0).unsqueeze(1), scale_factor=(2,2), mode="nearest")
                mask_param = mask_param.squeeze(0).squeeze(1)
                psf_module_vis = PhysicalLayerVisualization(setup_params, 0, 0, 1)

            # turn recovered positions into phases
            xyz_rec = np.expand_dims(xyz_rec, 0)
            phases_np = EmittersToPhases(xyz_rec, setup_params)
            phases_emitter_rec = complex_to_tensor(phases_np).to(device)

            # use a uniform number of photons for recovery visualization
            nphotons_rec = 5000 * torch.ones((1, xyz_rec.shape[1])).to(device)

            # generate the recovered image by the net
            exp_pred_im = psf_module_vis(mask_param, phases_emitter_rec, nphotons_rec)

            # compare the recovered image to the input
            ShowRecNetInput(exp_tensor, 'Experimental Input to Localization Net')
            ShowRecNetInput(exp_pred_im, 'Recovered Input Matching Net Localizations')

            # return recovered locations and net confidence
            return np.squeeze(xyz_rec, 0), conf_rec

        else:

            # ==========================================================================================================
            # create a data generator to efficiently load imgs for temporal acquisitions
            # ==========================================================================================================
            
            # sort images by number (assumed names are <digits>.tif)
            img_names = sort_names_tif(img_names)

            # instantiate the data class and create a data loader for testing
            num_imgs = len(img_names)
            exp_test_set = ExpDataset(img_names, setup_params)
            exp_generator = DataLoader(exp_test_set, batch_size=1, shuffle=False)

            # time the entire dataset analysis
            tall_start = time.time()

            # needed pixel-size for plotting if only few images are in the folder
            pixel_size_FOV = setup_params['pixel_size_FOV']

            # needed recovery pixel size and minimal axial height for turning ums to nms
            psize_rec_xy, zmin = setup_params['pixel_size_rec'], setup_params['zmin']

            # process all experimental images
            cnn.eval()
            results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
            with torch.set_grad_enabled(False):
                for im_ind, exp_im_tensor in enumerate(exp_generator):

                    # print current image number
                    print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

                    # time each frame
                    tfrm_start = time.time()

                    # transfer normalized image to device (CPU/GPU)
                    exp_im_tensor = exp_im_tensor.to(device)

                    # predicted volume using model
                    pred_volume = cnn(exp_im_tensor)

                    # post-process result to get the xyz coordinates and their confidence
                    xyz_rec, conf_rec = postprocessing_module(pred_volume)

                    # time it takes to analyze a single frame
                    tfrm_end = time.time() - tfrm_start

                    # if this is the first image, get the dimensions and the relevant center for plotting
                    if im_ind == 0:
                        _, _, H, W = exp_im_tensor.size()
                        ch, cw = np.floor(H / 2), np.floor(W / 2)

                    # if prediction is empty then set number fo found emitters to 0
                    # otherwise generate the frame column and append results for saving
                    if xyz_rec is None:
                        nemitters = 0
                    else:
                        nemitters = xyz_rec.shape[0]
                        frm_rec = (im_ind + 1)*np.ones(nemitters)
                        xyz_save = xyz_to_nm(xyz_rec, H*2, W*2, psize_rec_xy, zmin)
                        results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

                    # visualize the first 10 images regardless of the number of expeimental frames
                    visualize_flag = True if im_ind < 10 else (num_imgs <= 100)
                    
                    # if the number of imgs is small then plot each image in the loop with localizations
                    if visualize_flag:

                        # show input image
                        fig100 = plt.figure(100)
                        im_np = np.squeeze(exp_im_tensor.cpu().numpy())
                        imfig = plt.imshow(im_np, cmap='gray')
                        plt.plot(xyz_rec[:, 0] / pixel_size_FOV + cw, xyz_rec[:, 1] / pixel_size_FOV + ch, 'r+')
                        plt.title('Single frame complete in {:.2f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                        fig100.colorbar(imfig)
                        plt.draw()
                        plt.pause(0.05)
                        plt.clf()

                    else:

                        # print status
                        print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

            # print the time it took for the entire analysis
            tall_end = time.time() - tall_start
            print('=' * 50)
            print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
            print('=' * 50)

            # write the results to a csv file named "localizations.csv" under the exp img folder
            row_list = results.tolist()
            curr_dt = datetime.now()
            curr_date, curr_time = f'{curr_dt.day}{curr_dt.month}{curr_dt.year}', f'{curr_dt.hour}{curr_dt.minute}{curr_dt.second}'
            with open(exp_imgs_path + f'localizations' + '_' + curr_date + '_' + curr_time + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)

            # return the localization results for the last image
            return xyz_rec, conf_rec


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously trained model
    parser.add_argument('--path_results', help='path to the results folder for the pre-trained model', required=True)

    # previously trained model
    parser.add_argument('--postprocessing_params', help='post-processing dictionary parameters', required=True)

    # path to the experimental images
    parser.add_argument('--exp_imgs_path', default=None, help='path to the experimental test images')

    # seed to run model
    parser.add_argument('--seed', default=66, help='seed for random test data generation')

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    xyz_rec, conf_rec = test_model(args.path_results, args.postprocessing_params, args.exp_imgs_path, args.seed)
