import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt



def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def accuracy(y_predict, y):
    correct = 0
    for i, sample in enumerate(y):
        if sample == y_predict[i]:
            correct += 1
    return correct

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    test_setX = np.array([]).reshape(0,x.shape[1])
    train_setX = np.array([]).reshape(0,x.shape[1])
    test_setY = np.array([])
    train_setY = np.array([])
    for i in range(len(k_indices)):
      
        if i == k:
            test_setX = np.append(test_setX, x[k_indices[i]], 0)
            test_setY = np.append(test_setY, y[k_indices[i]])
            
        else:
            train_setX = np.append(train_setX, x[k_indices[i]], 0)
            train_setY = np.append(train_setY, y[k_indices[i]])
    #test_setX = build_poly(test_setX, degree)
    #train_setX = build_poly(train_setX, degree)
    
    
    w, mse = ridge_regression(train_setY, train_setX, lambda_)
    y_predict_test = predict_labels(w, test_setX)
    y_predict_train = predict_labels(w, train_setX)
    
    acc_te = accuracy(y_predict_test, test_setY)
    acc_tr = accuracy(y_predict_train, train_setY)
   
    return acc_tr, acc_te






def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_demo(y, tx, lambdas ):
    seed = 1
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for lambda_ in lambdas:
        tr_temp = 0
        te_temp = 0
        for k in range(k_fold):
            acc_tr, acc_te = cross_validation(y, tx, k_indices, k, lambda_)
            tr_temp += acc_tr
            te_temp += acc_te
        rmse_tr.append(tr_temp/k_fold)
        rmse_te.append(te_temp/k_fold)
    print(rmse_tr)
    print(rmse_te)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
