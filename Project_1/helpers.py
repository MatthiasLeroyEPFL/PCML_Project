# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt


def standardize(x):
    '''Standardize the original data set.'''
    std_x = np.std(x,0)
    mean_x = np.mean(x,0)
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x


def build_poly_cross(x, end, combination_types):
    ''' 
    Create new features as input
    First create polynomial basis of degree end
    Then concatenate with cross-terms [0], square cross-terms [1], cubic root cross-terms [2],
    absolute value of x [3] and square root cross-terms
    '''
    
    matrix_poly = np.array([]).reshape(x.shape[0],0)
    comb = combinations(x)
    for col in x.transpose():
        pol = np.array([col**d for d in range(1, end+1)]).transpose()
        matrix_poly = np.concatenate((matrix_poly,pol),1)
    
    if combination_types[2]:
        matrix_poly = np.concatenate((matrix_poly,np.cbrt(comb)), 1)
    if combination_types[0]:
        matrix_poly = np.concatenate((matrix_poly, comb), 1)
    if combination_types[1]:
        matrix_poly = np.concatenate((matrix_poly,np.square(comb)),1)
    if combination_types[3]:
        matrix_poly = np.concatenate((matrix_poly,np.abs(x)),1)
    
    matrix_poly = np.concatenate((matrix_poly,np.sqrt(np.abs(comb))),1)
    
    return matrix_poly


def add_ones(dataset):
    '''Add an offset term to the input.'''
    return np.concatenate((np.ones(dataset.shape[0]).reshape(-1,1), dataset), 1)


def combinations(dataset):
    '''Create cross-terms for the data.'''
    x, y = np.triu_indices(dataset.shape[1], 1)
    return dataset[:,x] * dataset[:,y]


def data_wrangling(jets_datasets, y_datasets, jets_datasets_test, y_datasets_test, 
                   column, deg, combination, square_combination, square_root_combination):
    '''Function which takes as parameters: 4 datasets, an index of a column, a polynomial degreee
        and 3 boolean values corresponding to 3 methods which expand our features. 
        Returns processed train and test datasets.'''
    
    data_train, mean_train, std_train = standardize(jets_datasets[column])
    y_train = y_datasets[column]
    
    data_test = (jets_datasets_test[column] - mean_train) / std_train
    y_test = y_datasets_test[column]
    
    data_train, mean_train, std_train = standardize(build_poly(data_train, deg, combination, square_combination, square_root_combination))
    data_test = (build_poly(data_test, deg, combination, square_combination, square_root_combination) - mean_train) / std_train
    
    data_train = add_ones(data_train)
    data_test = add_ones(data_test)
    
    return data_train, data_test, y_train, y_test


def indexes_by_features(features, features_test):
    '''We split our inputs into 6 smaller datasets 
       based on their features and return 2 arrays with corresponding indices'''
    #Train Indexes
    index_train = []
    index_train.append(np.where((features[:,22] == 0) & (features [:,0] == -999)))
    index_train.append(np.where((features[:,22] == 0) & (features [:,0] != -999)))

    index_train.append(np.where((features[:,22] == 1) & (features [:,0] == -999)))
    index_train.append(np.where((features[:,22] == 1) & (features [:,0] != -999)))

    index_train.append(np.where(((features[:,22] == 2) | (features[:,22] == 3)) & (features [:,0] == -999)))
    index_train.append(np.where(((features[:,22] == 2) | (features[:,22] == 3)) & (features [:,0] != -999)))

    #Test Indexes
    index_test = []
    index_test.append(np.where((features_test[:,22] == 0) & (features_test[:,0] == -999)))
    index_test.append(np.where((features_test[:,22] == 0) & (features_test[:,0] != -999)))

    index_test.append(np.where((features_test[:,22] == 1) & (features_test[:,0] == -999)))
    index_test.append(np.where((features_test[:,22] == 1) & (features_test[:,0] != -999)))

    index_test.append(np.where(((features_test[:,22] == 2) | (features_test[:,22] == 3)) & (features_test[:,0] == -999)))
    index_test.append(np.where(((features_test[:,22] == 2) | (features_test[:,22] == 3)) & (features_test[:,0] != -999)))
    
    return index_train, index_test


def create_dataset(dataset, y, index):
    '''Create and return 6 datasets based on indexes as parameters 
       and remove some columns with no additionnal information
       for the process of each dataset.'''
    jet0_nm = dataset[index[0]]
    y0_nm = y[index[0]]
    
    jet0_wm = dataset[index[1]]
    y0_wm = y[index[1]]
    
    jet1_nm = dataset[index[2]]
    y1_nm = y[index[2]]
    
    jet1_wm = dataset[index[3]]
    y1_wm = y[index[3]]
    
    jet2_nm = dataset[index[4]]
    y2_nm = y[index[4]]
    
    jet2_wm = dataset[index[5]]
    y2_wm = y[index[5]]
    
    jet0_nm = np.delete(jet0_nm, [0, 4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], 1)
    jet0_wm = np.delete(jet0_wm, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], 1)
    jet1_nm = np.delete(jet1_nm, [0, 4, 5, 6, 12, 22, 26, 27, 28], 1)
    jet1_wm = np.delete(jet1_wm, [4, 5, 6, 12, 22, 26, 27, 28], 1)
    jet2_nm = np.delete(jet2_nm, [0, 22], 1)
    jet2_wm = np.delete(jet2_wm, 22, 1)
    
    return [jet0_nm, jet0_wm, jet1_nm, jet1_wm, jet2_nm, jet2_wm], [y0_nm, y0_wm, y1_nm, y1_wm, y2_nm, y2_wm]

def preprocessed_dataset(train_sets, test_sets):
    preprocessed_test_sets = []
    preprocessed_train_sets = []
    for i, jet_set in enumerate(train_sets):
        jet_set, mean, std = standardize(jet_set)
        preprocessed_test_sets.append((test_sets[i] - mean) / std)
        preprocessed_train_sets.append(jet_set)
    return preprocessed_train_sets, preprocessed_test_sets

def add_features(train_sets, test_sets):
    train_sets[0] = add_features_jet0_nm(train_sets[0])
    test_sets[0] = add_features_jet0_nm(test_sets[0])
    
    train_sets[1] = add_features_jet0_wm(train_sets[1])
    test_sets[1] = add_features_jet0_wm(test_sets[1])
    
    train_sets[2] = add_features_jet1_nm(train_sets[2])
    test_sets[2] = add_features_jet1_nm(test_sets[2])
    
    train_sets[3] = add_features_jet1_wm(train_sets[3])
    test_sets[3] = add_features_jet1_wm(test_sets[3])
    
    train_sets[5] = add_features_jet2_wm(train_sets[5])
    test_sets[5] = add_features_jet2_wm(test_sets[5])
    
    return train_sets, test_sets
    
def build_poly_cross_datasets(train_sets, test_sets):
    final_train_sets = []
    final_test_sets = []
    degrees = [2, 3, 2, 4, 2, 3]
    cross_terms = [[False, False, True, True], [True, False, True, False], [False, False, True, True],
                   [True, False, True, True], [False, False, False, True], [True, True, True, True]]
    
    for i, jet_set in enumerate(train_sets):
        temp_set, mean, std = standardize(build_poly_cross(jet_set, degrees[i], cross_terms[i]))
        final_test_sets.append(add_ones((build_poly_cross(test_sets[i], degrees[i], cross_terms[i]) - mean) / std))
        final_train_sets.append(add_ones(temp_set))
    
    return final_train_sets, final_test_sets
    

def compute_pseudo_rapidity(dataset, indexA, indexB):
    return np.abs(dataset[:,indexA] - dataset[:,indexB])


def add_features_jet0_wm(dataset):
    index_tau = [9, 10, 11]
    index_lep = [12, 13, 14]
    index_met = [15, 16, 17]
    
    features_to_add = []
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 9, 12))
        
    features_to_add = np.array(features_to_add).transpose()
    
    return np.concatenate((dataset,features_to_add), 1)

def add_features_jet0_nm(dataset):
    index_tau = [8, 9, 10]
    index_lep = [11, 12, 13]
    index_met = [14, 15, 16]
    
    features_to_add = []
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 9, 12))
        
    features_to_add = np.array(features_to_add).transpose()
    
    return np.concatenate((dataset,features_to_add), 1)

def add_features_jet1_nm(dataset):
    index_tau = [8, 9, 10]
    index_lep = [11, 12, 13]
    index_met = [14, 15, 16]
    jet_leading = [17,18,19]
    features_to_add = []
    
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 9, 12))
    features_to_add.append(compute_pseudo_rapidity(dataset, 9, 18))
    features_to_add.append(compute_pseudo_rapidity(dataset, 12, 18))
    
    features_to_add = np.array(features_to_add).transpose()
    
    return np.concatenate((dataset,features_to_add), 1)


def add_features_jet1_wm(dataset):
    index_tau = [9, 10, 11]
    index_lep = [12, 13, 14]
    index_met = [15, 16, 17]
    jet_leading = [18,19,20]
    
    features_to_add = []
    
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 10, 13))
    features_to_add.append(compute_pseudo_rapidity(dataset, 10, 19))
    features_to_add.append(compute_pseudo_rapidity(dataset, 13, 19))
        
    features_to_add = np.array(features_to_add).transpose()
    
    return np.concatenate((dataset,features_to_add), 1)

def add_features_jet2_wm(dataset):
    index_tau = [13, 14, 15]
    index_lep = [16, 17, 18]
    index_met = [19, 20, 21]
    jet_leading = [22,23,24]
    jet_sub = [25,26,27]
    features_to_add = []
    
    
    features_to_add.append(compute_pseudo_rapidity(dataset, 14, 17))
    features_to_add.append(compute_pseudo_rapidity(dataset, 14, 23))
    features_to_add.append(compute_pseudo_rapidity(dataset, 14, 26))
    features_to_add.append(compute_pseudo_rapidity(dataset, 17, 23))
    features_to_add.append(compute_pseudo_rapidity(dataset, 17, 26))
    
    
    features_to_add = np.array(features_to_add).transpose()
    
    return np.concatenate((dataset,features_to_add), 1)


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

