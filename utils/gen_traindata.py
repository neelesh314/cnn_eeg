import os
import sys
sys.path.append('utils/')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
from eeg_loader import EEGDataset, Normalize_X_Channel, ToTensor
from eeg_helpers import compute_channel_mean_std
import pickle

def genTrain(data_dir, subject_list):
    ''' Function to generate training EEG data in the form of torch tensors for training in pytorch
        params:
            data_dir (str): parent route for folders containing preprocessed FIF data
            subject_list (list): list of subject 
        return: (pytorch training data)
            eeg_dataset: return and saved as a pickle object

    '''

    # read all the data once to compute mean and std and generate the transform
    mean_channel, std_channel = compute_channel_mean_std(data_dir, subject_list)


    # construct transforms
    transform = transforms.Compose([Normalize_X_Channel(mean_channel, std_channel),
                                    ToTensor()])
    # create dataset
    eeg_dataset = EEGDataset(data_dir_active, 
                                 subject_list,
                                 transform=transform)
    print('Usable Data Samples: ', len(eeg_dataset))
    with open('eeg_directions.pkl', 'wb') as output:
        pickle.dump(eeg_dataset, output, pickle.HIGHEST_PROTOCOL)
    return eeg_dataset


if __name__ == "__main__":
    
    DATA_DIR = './preprocessed/'
    SUBJECT_LIST = [i for i in range(13)]
    
    _ = genTrain(DATA_DIR, SUBJECT_LIST)
