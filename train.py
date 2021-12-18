import os
import sys
sys.path.append('utils/')
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
from eeg_helpers import train_validate_split_subjects, test_accuracy
import pickle
from network import ConvNet


def train_network(dataset=None, network=ConvNet,
                  validate_subject_list=[1],
                  lr=1e-3, batch_size=64, epoch=10, use_cuda=True):
    """
    Train 3D-CNN for Direction classification
    :param dataset: dataset class
    :param network: network class
    :param validate_subject_list: subject list for validation
    :param batch_size: batch size
    :param epoch: number of epochs for training
    :param use_cuda: if true use cuda
    :return: classification accuracies
    """
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    """
    2D Mapping array
    """
    eeg_map = np.array([[16, 11, 6, 7, 2, 0, 62, 63, 67, 73, 74, 79, 85], [21, 17, 12, 8, 4, 3, 1, 64, 66, 69, 69, 78, 84],
                    [26, 22, 18, 13, 9, 10, 5, 65, 68, 71, 72, 77, 83],
                    [35, 30, 27, 23, 19, 14, 15, 70, 75, 76, 82, 88, 94],
                    [40, 36, 31, 28, 24, 20, 80, 81, 87, 93, 98, 102, 103],
                    [45, 45, 41, 37, 32, 29, 25, 86, 92, 97, 101, 107, 108],
                    [52, 50, 46, 42, 38, 33, 92, 96, 100, 106, 110, 111, 112],
                    [55, 56, 57, 53, 47, 43, 39, 105, 109, 117, 115, 115, 116],
                    [59, 60, 61, 58, 54, 51, 48, 114, 119, 121, 120, 122, 120]])

    for sub in validate_subject_list:
        """"
        Initialize network
        """
        # Setup Network
        net = network()
        net = nn.DataParallel(net.to(device), device_ids=[0,1])
        net.train()

        """
        Initialize optimizers
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        """
        Initialize data loaders
        """
        print('Testing on subject: ', sub)
        val_subjects = [sub]  # subject for evaluation

        train_indices, val_indices = train_validate_split_subjects(eeg_dataset,
                                                                   val_subjects)

        # create samplers
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        val_sampler = sampler.SubsetRandomSampler(val_indices)
        
        # create dataloaders
        train_loader = DataLoader(eeg_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=train_sampler,
                                  num_workers=4,
                                  pin_memory=True)
        
        val_loader = DataLoader(eeg_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=val_sampler,
                                num_workers=4,
                                pin_memory=True)

        for epoch in range(epochs):
            print('EPOCH:', epoch)
            for i, data in enumerate(train_loader, 0):
                eeg, labels = data
                
                #map stacked EEG data onto topgraphy preserving EEG representation
                with torch.no_grad():
                    eeg = eeg.view(eeg.shape[0], eeg.shape[2], eeg.shape[3])
                    mapped_eeg = np.zeros((eeg.shape[0], 9, 13, eeg.shape[2]))
                    for l in range(eeg.shape[0]):
                        for j in range(9):
                            for k in range(13):
                                mapped_eeg[l, j, k, :] = eeg[l, eeg_map[j, k], :]
                    mapped_eeg = torch.from_numpy(mapped_eeg).view(
                        (mapped_eeg.shape[0], 1, mapped_eeg.shape[1], mapped_eeg.shape[2],
                         mapped_eeg.shape[3])).float().cuda()
                pred = net(mapped_eeg)
                """
                update the network
                """
                net.train()
                optimizer.zero_grad()
                loss = criterion(pred, labels.cuda())
                loss.backward()
                optimizer.step()
            
            net.eval()
            accuracy, class_accuracy = test_accuracy(net, val_loader)
            net.train()
            print('Overall all accuracy after epoch %d : %.3f %%\n' % (epoch + 1, accuracy * 100))
            print('Loss after epoch %d : %.3f %%\n' % (epoch + 1, loss))
            print('Accuracy for class 1: %.3f %% class 2: %.3f %% class 3: %.3f %% class 4: %.3f %%\n' % (class_accuracy[0] * 100, class_accuracy[1] * 100,
                class_accuracy[2] * 100, class_accuracy[3] * 100))



if __name__ == '__main__':
    """
    Load data
    """
    with open('eeg_directions.pkl', 'rb') as f:
        eeg_dataset = pickle.load(f)

    """
    Define All Training Parameters
    """
    BATCH_SIZE = 64
    LR = 0.0001
    EPOCHS = 50
    VAL_LIST = [i for i in range(13)]
    USE_CUDA = True

    train_network(dataset=eeg_dataset, network=ConvNet,
                  validate_subject_list=VAL_LIST,
                  lr=LR, batch_size=BATCH_SIZE, epoch=EPOCHS, use_cuda=USE_CUDA)