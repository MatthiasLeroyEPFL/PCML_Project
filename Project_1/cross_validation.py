import numpy as np
from implementations import *
from proj1_helpers import *
from helpers import *
import matplotlib.pyplot as plt



def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title('cross_validation')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('cross_validation')

def accuracy(y_predict, y):
    correct = 0
    for i, sample in enumerate(y):
        if sample == y_predict[i]:
            correct += 1
    return correct/len(y)

def cross_validation(y, x, k_indices, k, lambda_, w_initial):
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
    #test_setX = build_poly(test_setX, 1,degree)
    #train_setX = build_poly(train_setX,1, degree)
    w = w_initial
    #w, mse = logistic_regression(train_setY, train_setX, w_initial, 20, lambda_)
   
    w, mse = ridge_regression(train_setY, train_setX, lambda_)
    rmse_tr = np.sqrt(2*mse)
    rmse_te = np.sqrt(2*compute_mse(test_setY, test_setX, w))
    pred = predict_labels(w, test_setX)
    acc = accuracy(pred, test_setY)
    return rmse_tr, rmse_te, acc






def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_demo(y, tx, lambdas , degrees=None, w_initial=None):
    seed = 1
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
   
    rmse_tr = []
    rmse_te = []
    min_te = 10**10
    min_lambda = 0
    max_lambda = 0
    min_acc = 0
    
    for lambda_ in lambdas:
        acc_te = 0
        tr = 0
        err_tr = 0
        err_te = 0
        for k in range(k_fold):
            loss_tr, loss_te, acc = cross_validation(y, tx, k_indices, k, lambda_, w_initial)
            err_tr += loss_tr
            err_te += loss_te
            acc_te += acc
        if err_te/k_fold < min_te:
            min_te = err_te/k_fold
            min_lambda = lambda_
        if acc_te/k_fold > min_acc:
            min_acc = acc_te/k_fold
            max_lambda = lambda_
        rmse_tr.append(err_tr/k_fold)
        rmse_te.append(err_te/k_fold)
    print(min_te,min_lambda)
    print(min_acc,max_lambda)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
