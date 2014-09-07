# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


def LoadRawDataset():
    l_file = scipy.io.loadmat('multi_data/test_samples.mat')
    test_samples = l_file['test_samples']
    l_file = scipy.io.loadmat('multi_data/test_labels.mat')
    test_labels = l_file['test_labels']

    l_file = scipy.io.loadmat('multi_data/train_samples.mat')
    train_samples = l_file['train_samples']
    l_file = scipy.io.loadmat('multi_data/train_labels.mat')
    train_labels = l_file['train_labels']

    l_file = scipy.io.loadmat('multi_data/valid_samples.mat')
    valid_samples = l_file['valid_samples']
    l_file = scipy.io.loadmat('multi_data/valid_labels.mat')
    valid_labels = l_file['valid_labels']

    samples = np.concatenate((test_samples, train_samples, valid_samples), axis=0)
    labels = np.concatenate((test_labels, train_labels, valid_labels), axis=0)
    
    return samples, labels
    
    
pd.set_option('display.mpl_style', 'default')
samples, labels = LoadRawDataset()

df = pd.DataFrame(samples)
df['label'] = labels

df_mean = pd.DataFrame()
df_var = pd.DataFrame()
for i in range(labels.max() + 1):
    temp_mean = df[df['label']==i].mean(axis=0)
    temp_var = df[df['label']==i].var(axis=0)
    df_mean[i] = temp_mean
    df_var[i] = temp_var
    
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax2.set_xlabel("Electromagnetic bands")
ax2.set_ylabel("Mean responses per band")
ax1.set_ylabel("Variance per band")    
df_var.columns = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
                   'Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']

df_var[:-1].plot(lw=2, ax=ax1)
df_mean[:-1].plot(lw=2, ax=ax2, legend=False)
