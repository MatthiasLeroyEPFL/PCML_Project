from helpers import *
import numpy as np

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    num_user = train.shape[1]
    num_feature = item_features.shape[1]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_user, num_feature))

    for user, items in nz_user_itemindices:
        
        solve_A = item_features[items,:].T.dot(item_features[items,:]) + lambda_I
        solve_B = item_features[items,:].T * train[items, user]
        
        X = np.linalg.solve(solve_A, solve_B)
        
        updated_user_features[user,:] = X.squeeze(axis=1)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = train.shape[0]
    num_feature = user_features.shape[1]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_item, num_feature))

    for item, users in nz_item_userindices:
        
        solve_A = user_features[users,:].T.dot(user_features[users,:]) + lambda_I
        solve_B = user_features[users,:].T * train[item, users].T
        
        X = np.linalg.solve(solve_A, solve_B)
        updated_item_features[item,:] = X.squeeze(axis=1)
    return updated_item_features



def ALS(train, test, num_features, lambda_user, lambda_item):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters   
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("\nstart the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features.T, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    if test != None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_error(test, user_features.T, item_features, nnz_test)
        print("test RMSE after running ALS: {v}.".format(v=rmse))
        return item_features.dot(user_features.T), rmse
    
    return item_features.dot(user_features.T)