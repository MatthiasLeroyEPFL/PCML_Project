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

def cross_validation(y, x, k_indices, k, lambda_, degree):
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
    
    w, mse = ridge_regression(train_setY, train_setX, lambda_)
    rmse_tr = compute_rmse(train_setY, train_setX,w)
    rmse_te = compute_rmse(test_setY, test_setX,w)
    y_te = predict_labels(w, test_setX)
    y_tr = predict_labels(w, train_setX)
    acc_te = accuracy(y_te,test_setY)
    acc_tr = accuracy(y_tr, train_setY)
    return acc_tr, acc_te, rmse_tr, rmse_te






def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_demo(y, tx, lambdas ,degree):
    seed = 1
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
   
    acc_tr = []
    acc_te = []
    rmse_tr = []
    rmse_te = []
    max_te = 0
    min_te = 10**10
    min_lambda = 0
    max_lambda = 0
    for lambda_ in lambdas:
        te = 0
        tr = 0
        err_tr = 0
        err_te = 0
        for k in range(k_fold):
            pred_tr, pred_te, loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lambda_,degree)
            te += pred_te
            tr += pred_tr
            err_tr += loss_tr
            err_te += loss_te
        if te/k_fold > max_te:
            max_te = te/k_fold
            max_lambda = lambda_
        if err_te/k_fold < min_te:
            min_te = err_te/k_fold
            min_lambda = lambda_
        acc_tr.append(tr/k_fold)
        acc_te.append(te/k_fold)
        rmse_tr.append(err_tr/k_fold)
        rmse_te.append(err_te/k_fold)
    print(max_te, max_lambda)
    print(min_te,min_lambda)
    cross_validation_visualization(lambdas, acc_tr, acc_te)
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)
