# ======================================================================================================================
# Demo 2 demonstrates how to use a model trained on simulations to localize experimental data. Again, the experimental
# frames are taken from our STORM experiment (Fig. 3 main text). Note that the frames saved in the folder
# <Experimental_Data/Tetrapod_demo2/> are those after minimum subtraction.
# ======================================================================================================================

# import related script and packages to load the exp. image
from DeepSTORM3D.Testing_Localization_Model import test_model
import matplotlib.pyplot as plt
import os

# pre-trained weights on simulations
path_curr = os.getcwd()
path_results = path_curr + '/Demos/Results_Tetrapod_demo2/'

# path to experimental images
path_exp_data = path_curr + '/Experimental_Data/Tetrapod_demo2_crop/'

# postprocessing parameters
postprocessing_params = {'thresh': 40, 'radius': 4, 'keep_singlez': False}

# test the model by comparing the regenerated image alongside the 3D positions
xyz_rec, conf_rec = test_model(path_results, postprocessing_params, path_exp_data)

# show all plots
plt.show()


