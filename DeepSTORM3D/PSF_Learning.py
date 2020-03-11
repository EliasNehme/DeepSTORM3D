# ============================================================================
# CUDA FFT Training script
# ============================================================================

# Import modules and libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import numpy as np
import matplotlib.pyplot as plt
from DeepSTORM3D.data_utils import PhasesOnlineDataset
from DeepSTORM3D.cnn_utils import OpticsDesignCNN
from DeepSTORM3D.loss_utils import KDE_loss3D, jaccard_coeff
from DeepSTORM3D.helper_utils import save_checkpoint, resume_from_checkpoint
from DeepSTORM3D.vis_utils import ShowLossJaccardAtEndOfEpoch, ShowMaskPSF
import os
import time
import argparse


def learn_mask(setup_params):

    # set random number generators for repeatability
    torch.manual_seed(999)
    np.random.seed(526)

    # train on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True

    # training data folder for loading
    path_train = setup_params['training_data_path']

    # check whether this is the right training script and initialize the phase mask to be learned
    if setup_params['learn_mask'] is False:
        print('Wrong training script! You should use the function learn_localization_cnn in '
              'Training_Localization_Model.py instead')
        return
    elif setup_params['mask_init'] is None:
        mask_init = np.zeros((setup_params['Hmask'], setup_params['Wmask']))
    else:
        mask_init = setup_params['mask_init']
    mask_param = nn.Parameter(torch.from_numpy(mask_init).type(torch.FloatTensor).to(device))
    
    # phase term for PSF visualization
    vis_term, zvis = setup_params['vis_term'], setup_params['zvis']

    # output folder for results
    path_save = setup_params['results_path']
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)

    # save setup parameters in the results folder as well
    path_setup_params = path_save + 'setup_params.pickle'
    with open(path_setup_params, 'wb') as handle:
        pickle.dump(setup_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # open all locations pickle file
    path_pickle = path_train + 'labels.pickle'
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)

    # Parameters for data loaders (batch size is set to 1 since examples are generated 16 at a time)
    params_train = {'batch_size': 1, 'shuffle': True}
    params_valid = {'batch_size': 1, 'shuffle': False}

    # instantiate the data class and create a data loader for training
    partition = setup_params['partition']
    training_set = PhasesOnlineDataset(partition['train'], labels, setup_params)
    training_generator = DataLoader(training_set, **params_train)

    # instantiate the data class and create a data loader for validation
    validation_set = PhasesOnlineDataset(partition['valid'], labels, setup_params)
    validation_generator = DataLoader(validation_set, **params_valid)

    # build model and convert all the weight tensors to cuda()
    print('=' * 50)
    print('CNN architecture')
    print('=' * 50)
    cnn = OpticsDesignCNN(setup_params)
    cnn.to(device)

    # training parameters
    max_epochs, initial_learning_rate = setup_params['max_epochs'], setup_params['initial_learning_rate']
    steps_per_epoch = setup_params['ntrain_batches']

    # gap between validation and training loss
    gap_thresh = 1e-4

    # adam optimizer
    optimizer = Adam(list(cnn.parameters()) + [mask_param], lr=initial_learning_rate)

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)

    # loss function
    scaling_factor = setup_params['scaling_factor']
    criterion = KDE_loss3D(scaling_factor)

    # Model layers and number of parameters
    print(cnn)
    print("number of parameters: ", sum(param.numel() for param in cnn.parameters()))

    #%% Training procedure

    # enable interactive plotting throughout iterations
    plt.ion()

    # in case we want to continue from a checkpoint
    if setup_params['resume_training']:
        start_epoch = resume_from_checkpoint(cnn, optimizer, path_save + "checkpoint.pth.tar")
        end_epoch = setup_params['num_epochs_resume'] + start_epoch

        # load the saved mask
        mask_param = torch.load(path_save + 'mask_best_loss.pkl')

        # load all recorded metrics
        path_learning_results = path_save + 'learning_results.pickle'
        with open(path_learning_results, 'rb') as handle:
            learning_results = pickle.load(handle)

        # initialize validation set loss and jaccard
        valid_loss_prev, valid_JI_prev = np.min(learning_results['losses_valid']), np.max(
            learning_results['jaccard_valid'])

    else:

        # start from scratch
        start_epoch, end_epoch, num_epochs = 0, setup_params['max_epochs'], setup_params['max_epochs']

        # initialize the learning results dictionary
        learning_results = {'train_loss': [], 'train_jacc': [], 'valid_loss': [], 'valid_jacc': [],
                            'max_valid': [], 'sum_valid': [], 'steps_per_epoch': steps_per_epoch}

        # initialize validation set loss to be infinity and jaccard to be 0
        valid_loss_prev, valid_JI_prev = float('Inf'), 0.0

    # starting time of training
    train_start = time.time()

    # loop over epochs
    not_improve = 0
    for epoch in np.arange(start_epoch, end_epoch):
        
        # starting time of current epoch
        epoch_start_time = time.time()

        # print current epoch number
        print('='*50)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('='*50)

        # training phase
        cnn.train()
        train_loss = 0.0
        train_jacc = 0.0
        with torch.set_grad_enabled(True):
            for batch_ind, (phases_emitter, Nphotons, targets) in enumerate(training_generator):

                # transfer data to variable on GPU
                phases_emitter = phases_emitter.to(device)
                Nphotons = Nphotons.to(device)
                targets = targets.to(device)
                
                # squeeze batch dimension (already done in sim.)
                phases_emitter = phases_emitter.squeeze(0)
                targets = targets.squeeze(0)    
                Nphotons = Nphotons.squeeze(0)
                
                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = cnn(mask_param, phases_emitter, Nphotons)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # running statistics
                train_loss += loss.item()
                jacc_ind = jaccard_coeff(outputs/scaling_factor, targets)
                train_jacc += jacc_ind.item()

                # print training loss
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f\n' % (epoch+1,
                      num_epochs, batch_ind+1, steps_per_epoch, loss.item()))

                # plot mask each 500 iterations
                if batch_ind % 50 == 0 or (epoch < 4 and batch_ind % 5 == 0):
                    plt.figure(1, figsize=(15, 5))
                    ShowMaskPSF(mask_param, vis_term, zvis, epoch*steps_per_epoch + batch_ind)

        # calculate and print mean validation loss and jaccard
        mean_train_loss = train_loss/setup_params['ntrain_batches']
        mean_train_jacc = train_jacc/setup_params['ntrain_batches']
        print('Mean training loss: %.4f, Mean training jaccard: %.4f\n'
              %(mean_train_loss, mean_train_jacc))

        # record training loss and jaccard results
        learning_results['train_loss'].append(mean_train_loss)
        learning_results['train_jacc'].append(mean_train_jacc)
        
        # set random number generators for repeatability
        torch.manual_seed(999)
        np.random.seed(526)

        # validation phase
        cnn.eval()
        valid_loss = 0.0
        valid_jacc = 0.0
        with torch.set_grad_enabled(False):
            for batch_ind, (phases_emitter, Nphotons, targets) in enumerate(validation_generator):

                 # transfer data to variable on GPU
                phases_emitter = phases_emitter.to(device)
                Nphotons = Nphotons.to(device)
                targets = targets.to(device)
                
                # squeeze batch dimension (already done in sim.)
                phases_emitter = phases_emitter.squeeze(0)
                targets = targets.squeeze(0)
                Nphotons = Nphotons.squeeze(0)

                # forward
                optimizer.zero_grad()
                outputs = cnn(mask_param, phases_emitter, Nphotons)
                val_loss = criterion(outputs, targets)

                # running statistics
                valid_loss += val_loss.item()
                jacc_ind = jaccard_coeff(outputs/scaling_factor, targets)
                valid_jacc += jacc_ind.item()

        # calculate and print mean validation loss and jaccard
        mean_valid_loss = valid_loss/setup_params['nvalid_batches']
        mean_valid_jacc = valid_jacc/setup_params['nvalid_batches']
        print('Mean validation loss: %.4f, Mean validation jaccard: %.4f\n'
              %(mean_valid_loss, mean_valid_jacc))

        # record validation loss and jaccard results
        learning_results['valid_loss'].append(mean_valid_loss)
        learning_results['valid_jacc'].append(mean_valid_jacc)

        # reduce learning rate if loss stagnates
        scheduler.step(mean_valid_loss)
        
        # sanity check: record maximal value and sum of last validation sample
        max_last = outputs.max()
        sum_last = outputs.sum()/params_valid['batch_size']
        learning_results['max_valid'].append(max_last)
        learning_results['sum_valid'].append(sum_last)

        # saving checkpoint: save best model so far
        if mean_valid_loss < (valid_loss_prev - 1e-4):

            # print an update and save the model weights
            print('Mean Validation Loss Improved from %.4f to %.4f, Saving Model Weights...'
                  % (valid_loss_prev, mean_valid_loss))
            torch.save(cnn.state_dict(), path_save + 'weights_best_loss.pkl')
            torch.save(mask_param.data, path_save + 'mask_best_loss.pkl')

            # change minimal loss and init stagnation indicator
            valid_loss_prev = mean_valid_loss
            not_improve = 0

            # save model and current loss + optimizer
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': cnn.state_dict(),
                'best_loss': mean_valid_loss,
                'optimizer': optimizer.state_dict()},
                path_save + 'checkpoint.pth.tar')
        else:
            # update stagnation indicator
            not_improve += 1
            print('No improvement in mean loss for %d epochs' % not_improve)

        # save also the best model in terms of jaccard index
        if mean_valid_jacc > (valid_JI_prev + 1e-4):

            # print an update and save the model weights
            print('Mean Validation Jaccard Index Impoved from %.4f to %.4f, Saving Model Weights...'
                  % (valid_JI_prev, mean_valid_jacc))
            torch.save(cnn.state_dict(), path_save + 'weights_best_jaccard.pkl')
            torch.save(mask_param.data, path_save + 'mask_best_jaccard.pkl')

            # change maximal jaccard to current value
            valid_JI_prev = mean_valid_jacc

        # plot loss evolution so far
        plt.figure(2)
        ShowLossJaccardAtEndOfEpoch(learning_results, epoch)

        # print max and sum status
        print('Max test last: %.4f, Sum test last: %.4f' %(max_last, sum_last))
        
        # report time it takes the net to complete the epoch
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                epoch_time_elapsed // 3600, 
                np.floor((epoch_time_elapsed / 3600 - epoch_time_elapsed // 3600)*60), 
                epoch_time_elapsed % 60))

        # save all records for latter visualization
        path_learning_results = path_save + 'learning_results.pickle'
        with open(path_learning_results, 'wb') as handle:
            pickle.dump(learning_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if no improvement for more than 7 epochs break training
        if not_improve >= 7:
            break

    # measure time that took the model to train
    train_time_elapsed = time.time() - train_start
    print('='*50)
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            train_time_elapsed // 3600, 
            np.floor((train_time_elapsed/3600 - train_time_elapsed // 3600)*60), 
            train_time_elapsed % 60))

    # print a summary of best jaccard and test loss
    print('Best Validation Loss: {:4f}'.format(mean_valid_loss))
    print('Best Validation Jaccard: {:4f}'.format(mean_valid_jacc))

    # save training time and best loss and jaccard
    learning_results['epoch_converged'] = epoch
    learning_results['last_epoch_time'], learning_results['training_time'] = epoch_time_elapsed, train_time_elapsed
    learning_results['best_valid_loss'], learning_results['best_valid_jaccard'] = mean_valid_loss, mean_valid_jacc
    path_learning_results = path_save + 'learning_results.pickle'
    with open(path_learning_results, 'wb') as handle:
        pickle.dump(learning_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously wrapped settings dictionary
    parser.add_argument('--setup_params', help='path to the parameters wrapped in the script parameter_setting', required=True)

    # parse the input arguments
    args = parser.parse_args()

    # run the phase mask learning function
    learn_mask(args.setup_params)
