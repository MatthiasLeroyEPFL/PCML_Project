'''This file groups the 6 asked methods and some useful additionnal others'''

import numpy as np



'''------------ 6 Required Methods ------------'''

#Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
    return w, loss


#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    w = initial_w
    loss = 0
    for batch_y, batch_x in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_stoch_gradient(batch_y, batch_x, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient          
    return w, loss


#Least squares regression using normal equations
def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.transpose(),tx), np.dot(tx.transpose(), y))
    loss = compute_mse(y, tx, w)
    return w, loss


#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    temp = np.dot(tx.transpose(), tx) + (lambda_* 2 * tx.shape[0]) * np.identity(tx.shape[1])
    w = np.linalg.solve(temp, np.dot(tx.transpose(), y))
    loss = compute_mse(y,tx,w)
    return w, loss


#Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

        
#Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_newton_method(y, tx, lambda_, w, gamma)    
    return w, loss



'''------------ Other Additionnal Methods ------------'''


def compute_loss(y, tx, w):
    e = y - np.dot(tx,w)
    #loss = 1/(2*tx.shape[0]) * np.dot(np.transpose(e),e)
    
    #loss1 = 1/tx.shape[0] * np.sum(np.absolute(e)) 
    return e

def compute_mse(y, tx, w):
    e = compute_loss(y, tx, w)
    return 1/(2*tx.shape[0]) * np.dot(np.transpose(e),e)

def compute_mae(y, tx, w):
    e = compute_loss(y, tx, w)
    return 1/tx.shape[0] * np.sum(np.absolute(e))

def compute_rmse(y, tx, w):
    return np.sqrt(2 * compute_mse(y, tx, w))

def compute_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    gradient = -1/tx.shape[0] * np.dot(np.transpose(tx),e)
    return gradient

def sigmoid_scalar(t):
    if t > 0:
        return 1 / (1 + np.exp(-t))
    return np.exp(t) / (1 + np.exp(t))

sigmoid = np.vectorize(sigmoid_scalar)

def calculate_loss(y, tx, w):
    return np.sum(np.log(1 + np.exp(tx.dot(w))) - y.transpose().dot(tx.dot(w)))

def calculate_gradient(y, tx, w):
    temp = sigmoid(tx.dot(w)) - y
    return tx.transpose().dot(temp)

def calculate_hessian(y, tx, w):
    S = sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w)))
    S = np.diag(np.ravel(S))
    return tx.transpose().dot(S).dot(tx)

def learning_by_newton_method(y, tx, w, gamma):
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    
    w = w - gamma * np.linalg.inv(hessian).dot(gradient)
    return loss, w

def learning_by_gradient_descent(y, tx, w, gamma):
    
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
   
    loss = calculate_loss(y, tx, w) + lambda_ / 2 * np.linalg.norm(w)**2
    gradient = calculate_gradient(y, tx, w) + lambda_ * w * 2
    hessian = calculate_hessian(y, tx, w) + np.diag(2 * lambda_ * np.ones(calculate_hessian(y, tx, w).shape[0]))
    return loss, gradient, hessian

def learning_by_penalized_gradient(y, tx, lambda_, w, gamma ):
    
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    
    w = w - gamma * np.linalg.inv(hessian).dot(gradient)
    return loss, w

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
    
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    batch_size = 1
    w = initial_w
    loss = 0
    
    for batch_y, batch_x in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_stoch_gradient(batch_y, batch_x, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
                
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(np.dot(tx.transpose(),tx), np.dot(tx.transpose(), y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    temp = np.dot(tx.transpose(), tx) + (lambda_* 2 * tx.shape[0]) * np.identity(tx.shape[1])
    w = np.linalg.solve(temp, np.dot(tx.transpose(), y))
    
    loss = compute_mse(y,tx,w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
   
    threshold = 1e-8
    losses = []
    w = initial_w
   
    for iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

        
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w
    
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, lambda_, w, gamma)
        
    return w, loss

