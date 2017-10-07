import numpy as np

def compute_loss(y, tx, w):
    e = y - np.dot(tx,w)
    #loss = 1/(2*tx.shape[0]) * np.dot(np.transpose(e),e)
    
    #loss1 = 1/tx.shape[0] * np.sum(np.absolute(e)) 
    return loss

def compute_mse(y, tx, w):
    e = compute_loss(y, tx, w)
    return 1/(2*tx.shape[0]) * np.dot(np.transpose(e),e)

def compute_mae(y, tx, w):
    e = compute_loss(y, tx, w)
    return 1/tx.shape[0] * np.sum(np.absolute(e))

def compute_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    gradient = -1/tx.shape[0] * np.dot(np.transpose(tx),e)
    return gradient

def least_squares_GD(y, tx, initial_w, maw_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
    
    return w, loss


def least_squares_SGD(y, tx, initial_w, maw_iters, gamma):
    
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
    gram_matrix = np.linalg.inv(np.dot(tx.transpose(), tx))
    w = np.dot(gram_matrix, tx.transpose())
    w = np.dot(w, y)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    temp = np.dot(tx.transpose(), tx) + (lambda_* 2 * tx.shape[0]) * np.identity(tx.shape[1])
    inv = np.linalg.inv(temp)
    w = np.dot(np.dot(inv, tx.transpose()), y)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss
