'''
@Author: your name
@Date: 2020-03-27 15:52:48
@LastEditTime: 2020-03-30 18:39:11
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \LAB2\bp.py
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scipy.io import loadmat
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time

def Relu(x):
    x[x > 0] = x
    x[x <= 0] = 0
    return x


def LoadMatFile(dataset='mnist'):
    if dataset == 'usps':
        X = loadmat('usps_train.mat')
        X = X['usps_train']
        y = loadmat('usps_train_labels.mat')
        y = y['usps_train_labels']
    else:
        X = loadmat('mnist_train.mat')
        X = X['mnist_train']
        y = loadmat('mnist_train_labels.mat')
        y = y['mnist_train_labels']
    return X, y


loss = nn.CrossEntropyLoss()


class Recognition(nn.Module):
    def __init__(self, dim_input, dim_output, depth=3):
        super(Recognition, self).__init__()
        self.depth = depth
        self.linear = nn.Linear(dim_input, dim_input)
        self.final = nn.Linear(dim_input, dim_output)

        self.relu = nn.ReLU()

    def net(self):
        nets = nn.ModuleList()
        for i in range(self.depth-1):
            nets.append(self.linear)
            nets.append(self.relu)
        nets.append(self.final)
        return nets

    def forward(self, X):
        nets = self.net()
        for n in nets:
            X = n(X)
        y_pred = torch.sigmoid(X)
        return y_pred


def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array()
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def process_labels(y):
    labels = dict()
    for i in y:
        if i[0] in labels:
            continue
        else:
            labels[i[0]] = len(labels)

    Y = []
    for i in y:
        Y.append([labels[i[0]]])
    return np.array(Y), len(labels)


def eval(y_hat, y):
    y_hat = y_hat.detach().numpy()
    encoder = OneHotEncoder(categories='auto')
    y = encoder.fit_transform(y)
    y = y.toarray()
    roc = roc_auc_score(y, y_hat, average='micro')
    y_hat = props_to_onehot(y_hat)
    acc = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat, average='macro')

    recall = recall_score(y, y_hat, average='macro')
    return acc, precision, roc, recall


if __name__ == "__main__":
    data_name = 'usps'  # usps, mnist
    depth=2
    epoch=20
    lr=0.01
    batch_size=32
    test_size=0.2 #train: test = 8:2
    X, y = LoadMatFile(data_name)
    y, num_output = process_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net = Recognition(X_train.shape[1], num_output,depth=depth)
    optimzer = optim.SGD(net.parameters(), lr=lr)
    loss_history = []
    epoch_his = []
    acc_history = []
    roc_history = []
    recall_history = []
    precision_history = []
    start = time.time()


    for i in range(epoch):
        epoch_his.append(i)
        print('epoch ', i)
        net.train()
        for X, y in dataloader:
            X = Variable(X)
            y_pred = net(X)
            l = loss(y_pred, y.squeeze()).sum()
            optimzer.zero_grad()
            l.backward()
            optimzer.step()
        loss_history.append(l)
        net.eval()
        y_hat = net(X_test)
        acc, p, roc, recall = eval(y_hat, y_test)
        acc_history.append(acc)
        recall_history.append(recall)
        roc_history.append(roc)
        precision_history.append(p)
        print('loss:{}, acc:{}, precision:{}, roc:{}, recall:{}'.format(
            l, acc, p, roc, recall))
    elapsed = (time.time() - start)
    print('total time: {}'.format(elapsed))
    plt.plot(np.array(epoch_his), np.array(loss_history), label='loss')
    plt.legend()
    plt.savefig('loss_{}_depth{}_lr{}_epoch{}_batch{}.png'.format(data_name,depth,lr,epoch,batch_size))
    plt.show()

    plt.plot(np.array(epoch_his), np.array(acc_history), label='acc')
    plt.plot(np.array(epoch_his), np.array(
        precision_history), label='precision')
    plt.plot(np.array(epoch_his), np.array(roc_history), label='roc_auc')
    plt.plot(np.array(epoch_his), np.array(recall_history), label='recall')
    plt.legend()
    plt.savefig('metrics_{}_depth_lr{}_epoch{}_batch{}.png'.format(data_name,depth,lr,epoch,batch_size))
    plt.show()
