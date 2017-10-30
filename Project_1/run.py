''' This programs reproduce the prediction.csv file with the best score on Kaggle'''

import numpy as np
from proj1_helpers import *
from implementations import *
from helpers import *


print('Loading the datasets...')

'''Load train et test data'''
y, features, ids = load_csv_data('train.csv')
y_test , features_test, ids_test = load_csv_data('test.csv')

print('Building the 6 train sets and the 6 test sets...')

'''Create 2 arrays with indexes of interesting features'''
index_train, index_test = indexes_by_features(features, features_test)

'''Create train and test datasets based on previous indexes'''
jets_datasets, y_datasets = create_dataset(features, y, index_train) 
jets_datasets_test, y_datasets_test = create_dataset(features_test, y_test, index_test)


'''Preprocessed the 6 test sets and the 6 train sets'''
preprocessed_train_datasets, preprocessed_test_datasets  = preprocessed_dataset(jets_datasets, jets_datasets_test)

'''Add some features to the 12 datasets '''
train_datasets, test_datasets = add_features(preprocessed_train_datasets, preprocessed_test_datasets)

'''Build the polynomial and the cross terms for test and train'''

degrees = [2, 3, 2, 4, 2, 3]
cross_terms = [[False, False, True, True], [True, False, True, False], [False, False, True, True],
               [True, False, True, True], [False, False, False, True], [True, True, True, True]]
train_datasets, test_datasets = build_poly_cross_datasets(train_datasets, test_datasets, degrees, cross_terms)
print(train_datasets[5].shape)


print('Training the model...')

lambdas = [0.00908517575652, 3.72759372031*10**-7, 0.0177827941004, 0.000695192796178, 0.0215443469003, 0.00719685673001]
w = []
for i, jet_set in enumerate(train_datasets):
    w.append(ridge_regression(y_datasets[i], jet_set, lambdas[i])[0])

print('Making the predictions...')

y_predict = np.zeros(len(y_test))
for i, jet_set in enumerate(test_datasets):
    y_predict[index_test[i]] = predict_labels(w[i], jet_set)

create_csv_submission(ids_test, y_predict, 'prediction.csv')