from builtins import range
from matplotlib.pyplot import ScalarFormatter
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
      # np.reshape(X[i],(X[i].shape,1))
      score = X[i].dot(W)
      score_exp = np.exp(score)
      denom = np.sum(score_exp)
      true_class = y[i]
      prob = score_exp / denom
      # print(X[i].shape)
      loss += (-np.log(prob[y[i]]))
      for j in range(W.shape[1]):
        dW[:,j] += X[i] * prob[j]
      dW[:,y[i]] -= X[i]
      
    dW += 2*reg*W
    dW /= num_train
      
    loss /= num_train
    loss += reg*np.sum(W**2)



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score = X.dot(W)
    score_exp = np.exp(score)
    denom = np.sum(score_exp, axis = 1, keepdims = True)
    prob = score_exp / denom

    loss = -np.log(prob[np.arange(len(score)), y])
    loss = np.sum(loss)
    loss /= num_train
    loss += reg*np.sum(W**2)

    # dW = np.dot(X.T, prob)
    
    prob[np.arange(len(score)), y] -=1
    dW = np.dot(X.T, prob)
    # dW[:, y] = np.sum(np.dot(X.T, (prob-1)))
    dW += 2*reg*W
    dW /= num_train




    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
