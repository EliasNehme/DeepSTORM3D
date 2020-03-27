# ======================================================================================================================
# Demo 4 demonstrates how DeepSTORM3D can be used to learn an optimal PSF for high density. The parameters assumed for
# this demo match our telomere simulations (Fig. 4 main text).The script shows the phase mask and the corresponding PSF
# being updated over iterations. The mask was initialized in this demo to zero-modulation.
# ======================================================================================================================

# import the data generation and optics learning net functions
from Demos.parameter_setting_demo4 import demo4_parameters
from DeepSTORM3D.GenerateTrainingExamples import gen_data
from DeepSTORM3D.PSF_Learning import learn_mask

# specified training parameters
setup_params = demo4_parameters()

# generate training data
gen_data(setup_params)

# learn a localization cnn
learn_mask(setup_params)
