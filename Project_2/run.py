from helpers import *
from sgd import *
from als import *
from models import *
import numpy as np
import time

start = time.time()

print('Loading the data...')
ratings = load_data('data_train.csv')

all_predictions = []
#Get the target rates for the models using Surprise
to_predict = get_to_predict()

#Train all the models
print('Training with SGD...')
predict_SGD = matrix_factorization_SGD(ratings,None, 0.025, 20, 0.1, 0.016)
all_predictions.append(predict_SGD)

print('Training with ALS...')
predict_ALS = ALS(ratings,None, 3, 0.2, 0.9)
all_predictions.append(predict_ALS)

print('Training with global mean...')
predict_global = baseline_global_mean(ratings, None)
all_predictions.append(predict_global)

print('Training with SVD...')
predict_SVD = svd_surprise(ratings, None, to_predict, 10)
all_predictions.append(predict_SVD)

print('Training with user mean...')
predict_user = np.array(baseline_user_mean(ratings, None))
all_predictions.append(predict_user)

print('Training with item mean...')
predict_item = np.array(baseline_item_mean(ratings,None)).reshape(10000,1)
all_predictions.append(predict_item)

print('Training with KNN...')
predict_KNNitem = knn_surprise(ratings, None, to_predict, False)
all_predictions.append(predict_KNNitem)

print('Training with baseline...')
predict_baseline = baseline_surprise(ratings, None, to_predict)
all_predictions.append(predict_baseline)

print('Training with slope one...')
predict_slope = slope_surprise(ratings, None, to_predict)
all_predictions.append(predict_slope)

print('Mixing the models...')
weights = [ 0.65520667,  0.1487928 ,  0.04672615,  0.04749113,  0.08724932,
       -0.06340962,  0.40376256, -0.30433552, -0.01218448]

#Sum the weighted models
final_predict = 0
for i, pred in enumerate(all_predictions):
    final_predict += weights[i] * pred


final_predict[final_predict<1] = 1
final_predict[final_predict>5] = 5

print('Creating csv file...')
create_submissions(final_predict)

end = time.time()
print(end-start)

