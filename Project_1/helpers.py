# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt
from collections import Counter

def removeColumns(dataset):
    colToRemove = []
    for i, col in enumerate(dataset.transpose()):
        if -999 in col:
            colToRemove.append(i)
    return colToRemove
def compute_inverse_log(dataset, col_index):
  
    return np.log(1/(1+dataset[:,col_index]))

def clean_column(col):
    
    count = Counter(col)
    count.pop(-999)
    most_value = count.most_common(1)[0][0]
    col[col == -999] = np.nan
    mean = np.nanmean(col)
    index_train = np.where(np.isnan(col))
    col[index_train] = most_value
    #print(np.std(col))
    return col
    
def normalize(dataset):
    maximum = dataset.max(0)
    minimum = dataset.min(0)
    normalize_set = (dataset - minimum) / (maximum - minimum)
    return normalize_set
    
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

def standardize(x, mean_std=True):
    """Standardize the original data set."""
    std_x = np.std(x,0)
    mean_x = np.mean(x,0)
    x = x - mean_x
    
    x = x / std_x
    if mean_std:
        return x, mean_x, std_x
    return x

def build_poly(x, end, combination):
    matrix_poly = np.array([]).reshape(x.shape[0],0)
    for col in x.transpose():
        
        pol = np.array([col**d for d in range(1, end+1)]).transpose()
       
        matrix_poly = np.concatenate((matrix_poly,pol),1)
        
    if combination:
        return np.concatenate((matrix_poly, combinations(x)), 1)
    return matrix_poly

def add_ones(dataset):
    return np.concatenate((np.ones(dataset.shape[0]).reshape(-1,1), dataset), 1)

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
        for j in range(i+1,number_col):
            col_i = dataset[:,i]
            col_j = dataset[:,j]
        
            new_col = (col_i * col_j).reshape(dataset.shape[0],1)
            
            matrix = np.concatenate((matrix,new_col), 1)
    return matrix

def compute_momentum(tra_momentum, pseudo_rap, azi_angle):
    x = tra_momentum * np.cos(azi_angle)
    y = tra_momentum * np.sin(azi_angle)
    z = tra_momentum * np.sinh(pseudo_rap)
    return x, y, z

def compute_invariant_mass(dataset, columns_indexA, columns_indexB):
    a_x, a_y, a_z = compute_momentum(dataset[:, columns_indexA[0]], dataset[:, columns_indexA[1]], dataset[:, columns_indexA[2]])
    b_x, b_y, b_z = compute_momentum(dataset[:, columns_indexB[0]], dataset[:, columns_indexB[1]], dataset[:, columns_indexB[2]])
    
    a = np.sqrt(np.square(a_x) + np.square(a_y) + np.square(a_z))
    b = np.sqrt(np.square(b_x) + np.square(b_y) + np.square(b_z))
    
    invariant_mass = np.sqrt(np.square(a + b) - np.square(a_x + b_x) - np.square(a_y + b_y) - np.square(a_z + b_z))
    return invariant_mass


def compute_transverse_mass(dataset, columns_indexA, columns_indexB):
    a_x, a_y, a_z = compute_momentum(dataset[:, columns_indexA[0]], dataset[:, columns_indexA[1]], dataset[:, columns_indexA[2]])
    b_x, b_y, b_z = compute_momentum(dataset[:, columns_indexB[0]], dataset[:, columns_indexB[1]], dataset[:, columns_indexB[2]])
    
    a = np.sqrt(np.square(a_x) + np.square(a_y))
    b = np.sqrt(np.square(b_x) + np.square(b_y))
    
    transverse_mass = np.sqrt(np.square(a + b) - np.square(a_x + b_x) - np.square(a_y + b_y))
    return transverse_mass

def compute_pseudo_rapidity(dataset, indexA, indexB):
    return np.abs(dataset[:,indexA] - dataset[:,indexB])

def compute_product_pseudo(dataset, indexA, indexB):
    return dataset[:,indexA] * dataset[:,indexB]

def compute_r_separation(dataset, col_indexA, col_indexB):
    diff_pseudo = np.square(dataset[:,col_indexA[0]] - dataset[:,col_indexB[0]])
    diff_azi = dataset[:,col_indexA[1]] - dataset[:,col_indexB[1]]
    diff_azi[diff_azi > np.pi] = diff_azi[diff_azi > np.pi] - 2*np.pi
    diff_azi[diff_azi < -np.pi] = diff_azi[diff_azi < -np.pi] + 2*np.pi            
    return np.sqrt(diff_pseudo + np.square(diff_azi))

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

def add_features_jet2_nm(dataset):
    index_tau = [12, 13, 14]
    index_lep = [15, 16, 17]
    index_met = [18, 19, 20]
    index_jet_lead = [21, 22, 23]
    index_jet_sub = [24, 25, 26]
    
    features_to_add = []
   
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 16))
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 22))
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 25)) 
    features_to_add.append(compute_pseudo_rapidity(dataset, 16, 22))
    features_to_add.append(compute_pseudo_rapidity(dataset, 16, 25))
    features_to_add.append(compute_pseudo_rapidity(dataset, 22, 25))
    
    features_to_add.append(compute_r_separation(dataset, [13,14],[22,23]))
    features_to_add.append(compute_r_separation(dataset, [13,14],[25,26]))
    features_to_add.append(compute_r_separation(dataset, [16,17],[22,23]))
    features_to_add.append(compute_r_separation(dataset, [16,17],[25,26]))
    features_to_add.append(compute_r_separation(dataset, [22,23],[25,26]))
    
    
    features_to_add = np.array(features_to_add).transpose()
    
    
    return np.concatenate((dataset,features_to_add), 1)

def add_features_jet0(dataset):
    index_tau = [9, 10, 11]
    index_lep = [12, 13, 14]
    index_met = [15, 16, 17]
    
    features_to_add = []
    
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 9, 12))
        
    features_to_add = np.array(features_to_add).transpose()
    
    print(features_to_add.shape)
    return np.concatenate((dataset,features_to_add), 1)

def add_features_jet2_wm(dataset):
    index_tau = [13, 14, 15]
    index_lep = [16, 17, 18]
    index_met = [19, 20, 21]
    index_jet_lead = [22, 23, 24]
    index_jet_sub = [25, 26, 27]
    
    features_to_add = []
   
   
    
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 16))
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 22))
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 25)) 
    features_to_add.append(compute_pseudo_rapidity(dataset, 16, 22))
    features_to_add.append(compute_pseudo_rapidity(dataset, 16, 25))
    features_to_add.append(compute_pseudo_rapidity(dataset, 22, 25))
    
    
    features_to_add = np.array(features_to_add).transpose()
    #features_to_add, mean, std = standardize(features_to_add)
    
    print(features_to_add.shape)
    return np.concatenate((dataset,features_to_add), 1)
    