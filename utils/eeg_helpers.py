from __future__ import division
import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import mne
from eeg_loader import EEGDataset

def compute_channel_mean_std(data_dir, subject_list):
    ''' Function to compute cross channel EEG mean and std
        params:
            data_dir (str): parent route for folders contains FIF data
            subject_list (list): list of subject 
        return: (save 2 ndarray in data_dir)
            mean_channel (numpy ndarray): means for each channel
            std_channel (numpy ndarray): std for each channel
    '''
    # get data_w_label -> [data, label, subject, run]
    data_w_label = EEGDataset.read_raw_edf_datasets(data_dir,
            subject_list)
    data = data_w_label[0]

    # reshape data into [channel, time*number], connect all samples together
    data_reshape = np.zeros((data.shape[1], data.shape[0]*data.shape[2]))
    for num in range(data.shape[0]):
        data_reshape[:, num*data.shape[2]:(num+1)*data.shape[2]] = data[num, :, :]

    # compute mean
    mean_channel = np.mean(data_reshape, axis=1)
    # compute std
    std_channel = np.std(data_reshape, axis=1)
    # save mean and std
    np.save(data_dir+'mean.npy', mean_channel)
    np.save(data_dir+'std.npy', std_channel)
    return mean_channel, std_channel

def train_validate_split_subjects(dataset, val_subjects):
    ''' Function to split the dataset into Training and Validation sets
        Based on subjects
        params:
            dataset: A dataset 
            val_subjects: List of subjects we want to use for validation
        return:
            train_indices: List of indices for training data
            val_indices: List of indices for validation data
    '''
    subjects_data_list = dataset.data_w_label[2]
    train_indices = [i for i in range(len(dataset)) if not(subjects_data_list[i] in val_subjects)]
    val_indices = [i for i in range(len(dataset)) if subjects_data_list[i] in val_subjects]
    return train_indices, val_indices

def test_accuracy(network, test_data):
    '''Return the accuracy of the prediction of the network compares
        to the ground truth of test data.
        params:
            network: A pytorch network on cuda
            test_data: test/val dataloader
        return:
            overall_accuracy
            accuracy for each class
    '''
    class_correct = np.zeros(4)
    class_total = np.zeros(4)

    for data in test_data:
        eeg, labels = data
        eeg = eeg.float()
        eeg = eeg.view((eeg.shape[0], eeg.shape[2], eeg.shape[3]))

        #map the stacked EEG signals onto topography preserving plane 
        map = np.array([[16, 11, 6, 7, 2, 0, 62, 63, 67, 73, 74, 79, 85], [21, 17, 12, 8, 4, 3, 1, 64, 66, 69, 69, 78, 84],
                [26, 22, 18, 13, 9, 10, 5, 65, 68, 71, 72, 77, 83],
                [35, 30, 27, 23, 19, 14, 15, 70, 75, 76, 82, 88, 94],
                [40, 36, 31, 28, 24, 20, 80, 81, 87, 93, 98, 102, 103],
                [45, 45, 41, 37, 32, 29, 25, 86, 92, 97, 101, 107, 108],
                [52, 50, 46, 42, 38, 33, 92, 96, 100, 106, 110, 111, 112],
                [55, 56, 57, 53, 47, 43, 39, 105, 109, 117, 115, 115, 116],
                [59, 60, 61, 58, 54, 51, 48, 114, 119, 121, 120, 122, 120]])

        mapped_eeg = np.zeros((eeg.shape[0], 1,  9, 13, eeg.shape[2]))
        

        for l in range(eeg.shape[0]):
            for j in range(9):
                for k in range(13):
                    mapped_eeg[l,0,j,k,:] = eeg[l, map[j,k],:]

        mapped_eeg = torch.from_numpy(mapped_eeg).float().cuda()
        outputs = network(mapped_eeg)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.cpu() == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1.
    return class_correct.sum()/class_total.sum(), class_correct / class_total
        