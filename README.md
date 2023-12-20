# ml4science-synthetic-data

SYNTHETIC DATA GENERATION WITH VAE FOR BETTER MODEL FAIRNESS

This project aims to build a VAE model that allows to generate synthetic data to augment a dataset in order to increase a classifier's fairness. This classifier aims to predict whether a student would pass a test, based on their clicks profile on a virtual learning simulation.

The model can be run using one of four different oversampling methods, and the results of the final classifier' fairness and performance can be plotted.

----------------------------------------------------------------------------------------------------------
REQUIRED PACKAGES : (also found in the requirements.txt file)

imbalanced_learn==0.11.0
imblearn==0.0
ipython==8.17.2
keras==2.15.0
matplotlib==3.8.2
numpy==1.26.2
pandas==2.1.3
PyYAML==6.0.1
PyYAML==6.0.1
scikit_learn==1.3.2
tensorflow==2.15.0

----------------------------------------------------------------------------------------------------------
INSTRUCTIONS:

-To change the type of oversampling before running the code:
src/configs/config.yalm 
-> Change the 'rebalancing mode' between "rebalance", "100unbalanced", "100balanced", "5050balanced"
-> Change the 'number_categories' between "2cat", "4cat"

-To run the model in the terminal (go to src file with cd command):
Without any augmentation:
$ python script_oversample.py --mode baseline

With one of the augmentation type (written in the config file):
$ python script_oversample.py --mode augmentation

-To visualize the results:
src/configs/plotter_config.yalm
-> Change the type of metrics you want to plot (de-comment them)

For metrics plotting:
python script_plotter.py --nonnested --boxplot --show

For fairness plotting:
python script_plotter.py --nonnested --show --fairness --barplot --overall

----------------------------------------------------------------------------------------------------------
FILES:

data : contains the data file (ml4science_data_fake.pkl)(fake data for confidentiality reasons, so NOT the one that was used to train the model and plot the results)

experiments : contains the results of each run of the model

src : contains the code of the whole model
-> in src:

    -> in ml:
    
        -> in samplers:
        
            template_synthetic_oversampler : contains the code that runs to augment the data, whatever the augmentation mode
            
            final_utils : contains all the functions that template_synthetic_oversampler needs
            
            weights_files : contain the information of the best trained model that is used for the data augmentation as h5. files (using input dim = 23, batch size = 1, latent dim = 2, intermediate dim = 64, number of epochs = 120)
            
model_creation_training : contains the code that was used to create and train the VAE

best_parameters_search : contains the code that was used to plot the visualization of the latent space and find the best latent space and intermediate dimensions and number of epochs.


The files that we added or modified in the original github for our project are :
template_synthetic_oversampler (specifically the function _oversample and sample)
final_utils,
weights_files,
best_parameters_search,
model_creation_training,
config

----------------------------------------------------------------------------------------------------------
CREDITS:

Course: ML4Education, EPFL

Supervisors: Jade Maï Cock, Richard Lee Davis from the ML for Education laboratory

Students: Valentine Delevaux, Julie Le Tallec, Axel Croonenbroek
