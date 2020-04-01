'''
@Author: your name
@Date: 2020-03-30 18:46:54
@LastEditTime: 2020-04-01 19:10:31
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \LAB2\svm.py
'''
from sklearn.svm import SVC
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
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
    #标准化
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    return  X, y

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array()
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

def eval(y_hat, y):
    #y_hat = y_hat.detach().numpy()
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
    X,y=LoadMatFile('mnist')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    svc=SVC(kernel='rbf', class_weight='balanced',probability=True,gamma=0.01)

    y_train=np.reshape(y_train,(-1))
    svc.fit(X_train,y_train)

    svc.fit(X_train, y_train)
    y_hat=svc.predict_log_proba(X_test)

    print('score:{}'.format(svc.score(X_test,np.reshape(y_test,(-1)))))
    acc,precision,roc,recall=eval(y_hat,y_test)
    print('acc:{}, precision:{}, roc:{}, recall:{}'.format(acc,precision,roc,recall))
    