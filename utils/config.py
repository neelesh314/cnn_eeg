"""Preprocessing Variables"""
DROPPED_CHANNELS=['EXG2', 'EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']


"""BandPass Filter Params"""
LFREQ=0.1
HFREQ=40.

"""ICA params"""
N_COMPONENTS = 22


"""Event IDs for Direction classification"""
#event_id: 9 for left direction, 13 for right direction, 15 for up, 11 for down
L_EVENT=9
R_EVENT=13
U_EVENT=15
D_EVENT=11


"""Time slices for segmentation of EEG Epochs"""
TMIN=-0.5
TMAX=1.5

"""Resampling frequency"""
SMPL_FREQ=250 #Hz

