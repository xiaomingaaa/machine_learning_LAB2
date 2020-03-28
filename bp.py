'''
@Author: your name
@Date: 2020-03-27 15:52:48
@LastEditTime: 2020-03-28 15:03:38
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \LAB2\bp.py
'''
import torch
import torch.nn as nn
from scipy.io import loadmat


def Relu(x):
    x[x > 0] = x
    x[x <= 0] = 0
    return x
def LoadMatFile(dataset='mnist'):
    if dataset=='usps':
        X = loadmat('usps_train.mat')
        X=X['usps_train']
        y = loadmat('usps_train_labels.mat')
        y=y['usps_train_labels']
    else:
        X = loadmat('mnist_train.mat')
        X=X['mnist_train']
        y = loadmat('mnist_train_labels.mat')
        y=y['mnist_train_y']
    return X,y

if __name__ == "__main__":
    LoadMatFile('mnist')
