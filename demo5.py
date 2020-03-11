# ======================================================================================================================
# Demo 5 demonstrates how to use a model trained on simulations to localize an experimental image of telomeres within a
# U2OS cell nucleus. The PSF in the experimental data is the learned PSF (Fig. 5 main text). For visual comparison the
# script also regenerate the input image depending on the CNN localizations.
# ======================================================================================================================

# import related script and packages to load the exp. image
from DeepSTORM3D.Testing_Localization_Model import test_model
import matplotlib.pyplot as plt
import os

# pre-trained weights on simulations
path_curr = os.getcwd()
path_results = path_curr + '/Demos/Results_Learned_demo5/'

# path to experimental image (uncomment the example you want)
# path_exp_data = path_curr + '/Experimental_Data/Learned_demo5_frm1/'
path_exp_data = path_curr + '/Experimental_Data/Learned_demo5_frm2/'

# postprocessing parameters
postprocessing_params = {'thresh': 80, 'radius': 4}

# test the model by comparing the regenerated image alongside the 3D positions
xyz_rec, conf_rec = test_model(path_results, postprocessing_params, path_exp_data)

# show all plots
plt.show()
