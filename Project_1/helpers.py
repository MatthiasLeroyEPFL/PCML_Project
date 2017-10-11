# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt

def removeColumns(dataset):
    colToRemove = []
    for i, col in enumerate(dataset.transpose()):
        if -999 in col:
            colToRemove.append(i)
    return colToRemove

            
def cleanDataSet(dataTrain, dataTest):
    dataTrain[dataTrain == -999] = np.nan
    means = np.nanmean(dataTrain, axis = 0)
    indexTrain = np.where(np.isnan(dataTrain))
    dataTrain[indexTrain]=np.take(means,indexTrain[1])
    dataTrain, mean, std = standardize(dataTrain)
    
    dataTest[dataTest == -999] = np.nan
    indexTest = np.where(np.isnan(dataTest))
    dataTest[indexTest] = np.take(means, indexTest[1])
    dataTest = (dataTest - mean) / std
    return dataTrain, dataTest

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, start, end):
    matrix_poly = np.array([]).reshape(x.shape[0],0)
    for col in x.transpose():
        
        pol = np.array([col**d for d in range(start, end)]).transpose()
       
        matrix_poly = np.concatenate((matrix_poly,pol),1)
        
    return np.concatenate((x, matrix_poly), 1)
    
def split_data(x, y, ratio, seed=1):
   
    # set seed
    np.random.seed(seed)
    
    size = x.shape[0]
    all_index = np.arange(size)
    train_size = int(ratio * size)
    
    train_index = np.random.choice(all_index, train_size, False)
    test_index = np.setdiff1d(np.arange(x.shape[0]),train_index)
    training_setX = x[train_index]
    testing_setX = x[test_index]
    training_setY = y[train_index]
    testing_setY = y[test_index]
    
    return training_setX, testing_setX, training_setY, testing_setY, test_index         

def combinations(dataset):
    number_col = dataset.shape[1]
    matrix = np.array([]).reshape(dataset.shape[0],0)
    for i in range(number_col):
        for j in range(i,number_col):
            col_i = dataset[:,i]
            col_j = dataset[:,j]
        
            new_col = (col_i * col_j).reshape(dataset.shape[0],1)
            
            matrix = np.concatenate((matrix,new_col), 1)
    return matrix


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
