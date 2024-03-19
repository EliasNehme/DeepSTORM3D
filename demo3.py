# ======================================================================================================================
# Demo 3 demonstrates how to use a model trained on simulations to localize a simulated example with non-uniform
# background and emitter size. The parameters assumed for this demo match our Telomere simulations (Fig. 4 main text).
# This script also prints quantitative metrics and compare the result to the GT simulated positions.
# ======================================================================================================================

# test a Tetrapod pre-trained model
from DeepSTORM3D.Testing_Localization_Model import test_model
import matplotlib.pyplot as plt
import os

# path to the learning results
path_curr_dir = os.getcwd()
path_results = path_curr_dir + '/Demos/Results_Tetrapod_demo3/'

# set the postprocessing parameters
postprocessing_params = {'thresh': 80, 'radius': 4, 'keep_singlez': False}

# model testing
seed = 11  # you can change this to randomize the sampled example
test_model(path_results, postprocessing_params, None, seed)

# show all plots
plt.show()



