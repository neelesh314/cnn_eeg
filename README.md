# A 3D-Convolutional Neural Network to Decode Direction of Movement from EEG

This package is the PyTorch implementation of a 3D-CNN that classifies movement direction from the EEG signals. 

# Prerequisites

* Ubuntu 16.04
* Python 3.7
* PyTorch 1.0 (with CUDA 10.0)
* Python MNE

A CUDA enabled GPU is not required but preferred for training. 

## Example Usage ##

#### 1. EEG Preprocessing ####
The raw EEG data collected from the experiments need to be preprocessed first to get rid of common noise and artifacts. 
The preprocessing script is contained in 'eeg_preprocess.ipynb' and implements the following preprocessing steps: 
* Dropping bad channels from the EEG data through visual inspection
* Notch filter to remove power line noise
* Bandpass filter to remove slow drifts and high frequency noise
* Ocular artifact correction using ICA

The preprocessed data are saved as FIF files

#### 2. Generate Training Data ####
The preprocessed data is then used to create training data suitable for PyTorch. This is done through the function 'genTrain' contained in 'gen_traindata.py'.
The function takes in preprocessed data from the specified subjects, performs segmentations to extract events of interest (e.g. EEG epochs corresponding to different directions), and stores the EEG epochs as torch tensors. 

### 3. Build the Network and Train it ###
The training function is defined in 'train.py'. The function takes in the training dataset saved as a result of step 2, the network defined in 'network.py', list of validation subjects (as per leave-one-subject-out evaluation) and training details (batch size, learning rate). The function trains the network to optimize the classification loss corresponding to direction classification
