import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import mne
import numpy as np
import warnings
from config import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

class EEGDataset(Dataset):
    def __init__(self, data_dir, subject_list, transform=None):
        '''
        Init function for dataset class

        Args:
            data_dir (str): parent route for folders containing preprocessed FIF data
            subject_list (list): list of subjects
            transform (callable, optional): optional transform to be applied on a sample
        Init params:
            data_w_label(list): list of eeg_sample, label, subject
        '''
        self.data_w_label = self.read_raw_edf_datasets(data_dir,
                    subject_list)
        self.transform = transform
    
    def __len__(self):
        return self.data_w_label[0].shape[0]
    
    def __getitem__(self, idx):
        label = self.data_w_label[1][idx]
        eeg_sample = self.data_w_label[0][idx, :, :].astype('float64').reshape(123, -1, 1)
        sample_w_label = [eeg_sample, label]
        if self.transform:
            sample_w_label[0] = self.transform(sample_w_label[0])
        return sample_w_label
   
    @staticmethod    
    def read_raw_edf_datasets(data_dir, subject_list):
        '''
        Read raw EDF data function for dataset class

        Args:
            data_dir (str): parent route for folders containing preprocessed FIF data
            subject_list (list): list of subject 
        Return:
            data_w_label: [data, label, subject]
        '''
        overall_label = []
        overall_data = np.zeros(0)
        overall_subject = []
        
        # start reading data
        for sub_num in subject_list:
            sub_str = str(sub_num)
            
            fif_path = data_dir+'/sub'+sub_str+'.fif'
            
            # read edf using mne
            raw_data = mne.io.read_raw_fif(fif_path, preload=True)
               
            ch_names=['A1','A2','A3', 'A4','A5','A6','A7' ,'A8','A9','A10','A11', 'A12','A13','A14','A15' ,'A16','A17' ,'A18' ,'A19' ,'A20','A21' ,'A22',
        'A23' ,'A24','A25' ,'A26','A27','A28' ,'A29','A30' ,'A31' ,'A32' ,'B1','B2' ,'B3','B4' ,'B5','B6','B7' ,'B8','B9','B10', 'B11' ,'B12','B13' ,'B14' ,
        'B16' ,'B17','B18' ,'B19','B20' ,'B21','B22' ,'B23','B24','B25','B26','B27' ,'B28','B29' ,'B30','B31','C1','C2','C3','C4' ,'C5' ,'C6','C7' ,'C8' ,'C9','C10','C11' ,
         'C13','C14','C15' ,'C16' ,'C17' ,'C18','C19' ,'C20','C21' ,'C22','C23' ,'C24' ,'C25' ,'C26','C27','C28' ,'C29','C30' ,
        'C31','C32','D1','D2','D3','D4' ,'D5' ,'D6','D7' ,'D8','D9','D10', 'D11','D12' ,'D13',
        'D14','D15','D16' ,'D17','D18','D19' ,'D20','D21' ,'D22','D23' ,'D24' ,'D25','D27','D28' ,'D29' ,'D31' ,'D32','Status']
            
            raw_data = raw_data.reorder_channels(ch_names)
            
            #extract trigger events from the raw data
            raw_events = mne.find_events(raw_data, stim_channel='Status', consecutive=True, shortest_event=0, min_duration=0, mask_type='not_and', mask=16776960, uint_cast=True)
            
            #segment data based on the trigger events- event_id: 9 for left direction, 13 for right direction, 15 for up, 11 for down
            left_movement_epochs = mne.Epochs(raw_data,
                            events=raw_events,
                            event_id=[L_EVENT],
                            tmin=TMIN,
                            tmax=TMAX,
                            baseline=None,
                            picks=[i for i in range(123)], preload = True)
            left_movement_epochs.resample(SMPL_FREQ, npad = 'auto')
            left_movement_epochs_data = left_movement_epochs.get_data()
                
            right_movement_epochs = mne.Epochs(raw_data,
                            events=raw_events,
                            event_id=[R_EVENT],
                            tmin=TMIN,
                            tmax=TMAX,
                            baseline=None,
                            picks=[i for i in range(123)], preload = True)
            right_movement_epochs.resample(SMPL_FREQ, npad = 'auto')
            right_movement_epochs_data = right_movement_epochs.get_data()
            
            up_movement_epochs = mne.Epochs(raw_data,
                            events=raw_events,
                            event_id=[U_EVENT],
                            tmin=TMIN,
                            tmax=TMAX,
                            baseline=None,
                            picks=[i for i in range(123)], preload = True)
            up_movement_epochs.resample(SMPL_FREQ, npad = 'auto')
            up_movement_epochs_data = up_movement_epochs.get_data()
            
            down_movement_epochs = mne.Epochs(raw_data,
                            events=raw_events,
                            event_id=[D_EVENT],
                            tmin=TMIN,
                            tmax=TMAX,
                            baseline=None,
                            picks=[i for i in range(123)], preload = True)
            down_movement_epochs.resample(SMPL_FREQ, npad = 'auto')
            down_movement_epochs_data = down_movement_epochs.get_data()
          
            ##################################################
            # generate labels for this subject: 0 for left, 1 for right, 2 for up, 3 for down
            left_movement_epochs_label = [0 for i in range(left_movement_epochs_data.shape[0])]
            right_movement_epochs_label = [1 for i in range(right_movement_epochs_data.shape[0])]
            up_movement_epochs_label = [2 for i in range(up_movement_epochs_data.shape[0])]
            down_movement_epochs_label = [3 for i in range(down_movement_epochs_data.shape[0])]
            
            # generate subject list for this subject
            epochs_subject = [sub_num for i in range(left_movement_epochs_data.shape[0]+right_movement_epochs_data.shape[0]
                                            +up_movement_epochs_data.shape[0]+down_movement_epochs_data.shape[0])]

            #Concatenate to overall data, overall label, and overall subject list
            if overall_data.shape[0] == 0:
                overall_data = np.append(left_movement_epochs_data, right_movement_epochs_data, axis=0)
                overall_data = np.append(overall_data, up_movement_epochs_data, axis=0)
                overall_data = np.append(overall_data, down_movement_epochs_data, axis=0)
            
            else:
                tmp_2_epochs_data = np.append(left_movement_epochs_data, right_movement_epochs_data, axis=0)
                tmp_2_epochs_data = np.append(tmp_2_epochs_data, up_movement_epochs_data, axis=0)
                tmp_2_epochs_data = np.append(tmp_2_epochs_data, down_movement_epochs_data, axis=0)
                
                overall_data = np.append(overall_data, tmp_2_epochs_data, axis=0)
            
            overall_label += (left_movement_epochs_label + right_movement_epochs_label + up_movement_epochs_label + down_movement_epochs_label)
            overall_subject += epochs_subject
        # return
        return [overall_data, overall_label, overall_subject]
        

# EEG dataset transform classes
class Normalize_X_Channel(object):
    ''' Normalize the EEG data across channels given mean and dev '''
    def __init__(self, mean_channel, dev_channel):
        ''' init function
            params:
                mean_channel: 1D numpy array, mean of all EEG channels
                dev_channel: 1D numpy array, std of all EEG channels
        '''
        self.mean = mean_channel
        self.dev = dev_channel
    
    def __call__(self, sample):
        ''' __call__ function
            params:
                sample: [channel, time, 1]
        '''
        tmp_mean = self.mean.reshape(-1, 1, 1)
        tmp_dev = self.dev.reshape(-1, 1, 1)
        sample = (sample - tmp_mean) / tmp_dev
        return sample
        
class ToTensor(object):
    ''' Convert ndarray in sample to Tensors '''
    def __call__(self, sample):
        ''' __call__ function
            params:
                sample: numpy ndarray
        '''
        sample = sample.transpose((2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample
        
        
        
    
        
        