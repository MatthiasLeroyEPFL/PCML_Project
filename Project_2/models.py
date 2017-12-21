import scipy.sparse as sp
import numpy as np
from helpers import *
import pandas as pd
from surprise import *
import surprise


"""baseline method: use the user means as the prediction."""
def baseline_user_mean(train, test):
    
    mse = 0
    num_items, num_users = train.shape
    means = []

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
            means.append(user_train_mean)
        else:
            continue
        
        if test != None:
            # find the non-zero ratings for each user in the test dataset
            test_ratings = test[:, user_index]
            nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
            mse += calculate_mse(nonzeros_test_ratings, user_train_mean)
    if test != None:
        rmse = np.sqrt(1.0 * mse / test.nnz)
        print("test RMSE of the baseline using the user mean: {v}.".format(v=rmse))
    return means

"""baseline method: use the global mean."""
def baseline_global_mean(train, test):
    
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    if test != None:
        # find the non zero ratings in the test
        nonzero_test = test[test.nonzero()].todense()

        # predict the ratings as global mean
        mse = calculate_mse(nonzero_test, global_mean_train)
        rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
        print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))
    return global_mean_train
    
"""baseline method: use item means as the prediction."""    
def baseline_item_mean(train, test):
    
    mse = 0
    num_items, num_users = train.shape
    
    means = []
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
            means.append(item_train_mean)
        else:
            continue
        
        if test != None:
            # find the non-zero ratings for each movie in the test dataset
            test_ratings = test[item_index, :]
            nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
            # calculate the test error 
            mse += calculate_mse(nonzeros_test_ratings, item_train_mean)
    
    if test != None:
        rmse = np.sqrt(1.0 * mse / test.nnz)
        print("test RMSE of the baseline using the item mean: {v}.".format(v=rmse))
    return means

"""Format the data to use with Surprise"""    
def create_data(dataset):
    dataset_matrix = pd.DataFrame(dataset.todense())
    dataset_df = pd.DataFrame(dataset_matrix.unstack())
    dataset_df = dataset_df.reset_index() 
    dataset_df.rename(columns={'level_0': 'col', 'level_1': 'row', 0: 'rate'}, inplace=True)
    dataset_df['col'] +=1
    dataset_df['row'] +=1
    return dataset_df[dataset_df['rate'] != 0]    
    

"""Predictions using SVD algorithm of Surprise"""    
def svd_surprise(train, test, target, factors=20):
    
    train_df = create_data(train)
    reader = surprise.dataset.Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(train_df[['row', 'col', 'rate']], reader)
    data.split(2) 
    
    algo = SVD(n_factors=factors)
    algo.train(data.build_full_trainset())
    
    test_df = train_df.copy()
    temp = []
    if test != None:
        test_df = create_data(test)
    for index, row in test_df.iterrows():
        temp.append((int (row['row']), int (row['col']), row['rate']))
    if target == None:
        predictions = algo.test(temp)
    else:
        predictions = algo.test(target)
    final_predictions = []
    
    for pred in predictions:
        final_predictions.append((pred[0], pred[1], pred[3]))
    pred_df = pd.DataFrame(preprocess_data(final_predictions, True).todense())
    if test != None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_mix_error(data=test, prediction=pred_df.as_matrix(), nz=nnz_test)
        print(rmse)
        return pred_df.as_matrix(), rmse
    return pred_df.as_matrix()


"""Predictions using KNN algorithm of Surprise"""  
def knn_surprise(train, test, target, user_based=True, k=100):
    
    train_df = create_data(train)
    reader = surprise.dataset.Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(train_df[['row', 'col', 'rate']], reader)
    data.split(2) 
    
    sim_options = {'name': 'pearson_baseline',
               'user_based': user_based  
               }
    
    algo = KNNBaseline(k=k,sim_options=sim_options)
    algo.train(data.build_full_trainset())
    
    test_df = train_df.copy()
    temp = []
    if test != None:
        test_df = create_data(test)
        for index, row in test_df.iterrows():
            temp.append((int (row['row']), int (row['col']), row['rate']))
    if target == None:
        predictions = algo.test(temp)
    else:
        predictions = algo.test(target)
    final_predictions = []
    for pred in predictions:
        final_predictions.append((pred[0], pred[1], pred[3]))
    pred_df = pd.DataFrame(preprocess_data(final_predictions, True).todense())
    if test != None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_mix_error(data=test, prediction=pred_df.as_matrix(), nz=nnz_test)
        print(rmse)
        return pred_df.as_matrix(), rmse
    return pred_df.as_matrix()


"""Predictions using baseline algorithm of Surprise"""  
def baseline_surprise(train, test, target):
    
    train_df = create_data(train)
    reader = surprise.dataset.Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(train_df[['row', 'col', 'rate']], reader)
    data.split(2) 
    
    
    algo = BaselineOnly()
    algo.train(data.build_full_trainset())
    
    test_df = train_df.copy()
    temp = []
    if test != None:
        test_df = create_data(test)
        for index, row in test_df.iterrows():
            temp.append((int (row['row']), int (row['col']), row['rate']))
    if target == None:
        predictions = algo.test(temp)
    else:
        predictions = algo.test(target)
    final_predictions = []
    for pred in predictions:
        final_predictions.append((pred[0], pred[1], pred[3]))
    pred_df = pd.DataFrame(preprocess_data(final_predictions, True).todense())
    if test != None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        print(compute_mix_error(data=test, prediction=pred_df.as_matrix(), nz=nnz_test))
    return pred_df.as_matrix()


"""Predictions using slope one algorithm of Surprise"""  
def slope_surprise(train, test, target):
    
    train_df = create_data(train)
    reader = surprise.dataset.Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(train_df[['row', 'col', 'rate']], reader)
    data.split(2) 
    
    
    algo = SlopeOne()
    algo.train(data.build_full_trainset())
    
    test_df = train_df.copy()
    temp = []
    if test != None:
        test_df = create_data(test)
        for index, row in test_df.iterrows():
            temp.append((int (row['row']), int (row['col']), row['rate']))
    if target == None:
        predictions = algo.test(temp)
    else:
        predictions = algo.test(target)
    final_predictions = []
    for pred in predictions:
        final_predictions.append((pred[0], pred[1], pred[3]))
    pred_df = pd.DataFrame(preprocess_data(final_predictions, True).todense())
    if test != None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        print(compute_mix_error(data=test, prediction=pred_df.as_matrix(), nz=nnz_test))
    return pred_df.as_matrix()