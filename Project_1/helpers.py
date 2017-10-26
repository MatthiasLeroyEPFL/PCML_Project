# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt



#Standardize the original data set.
def standardize(x):
    std_x = np.std(x,0)
    mean_x = np.mean(x,0)
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x



def build_poly(x, end, combination, square_combination, square_root_combination):
    ''' 
    Create new features as input
    First create polynomial basis of degree end
    Then concatenate with cross-terms, square cross-terms and square root cross-terms'''
    
    matrix_poly = np.array([]).reshape(x.shape[0],0)
    comb = combinations(x)
    for col in x.transpose():
        pol = np.array([col**d for d in range(1, end+1)]).transpose()
        matrix_poly = np.concatenate((matrix_poly,pol),1)
        
    if combination:
        matrix_poly = np.concatenate((matrix_poly, comb), 1)
    if square_combination:
        matrix_poly = np.concatenate((matrix_poly,np.square(comb)),1)
    if square_root_combination:
         matrix_poly = np.concatenate((matrix_poly,np.sqrt(np.abs(comb))),1)
    return matrix_poly

#Add an offset term to the input
def add_ones(dataset):
    return np.concatenate((np.ones(dataset.shape[0]).reshape(-1,1), dataset), 1)

#Create cross-terms for the data
def combinations(dataset):
    x, y = np.triu_indices(dataset.shape[1], 1)
    return dataset[:,x] * dataset[:,y]


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

