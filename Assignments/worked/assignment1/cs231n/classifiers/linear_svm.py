import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # dW will be an array of shape (3073, 10) same as W
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # X is of shape (500, 3073), W is of shape (3073, 10)
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # SVM Loss function is Sum(Max(0, W[j].T.dot(X[i]) - W[y[i]].T.dot(X[i]) + delta))
                # Since there is max function, derivative applicable only if Sum > 0
                # Partial derivative w.r.t W[y[i]] is substraction from dW
                # Partial derivative w.r.t W[j] is addition to dW
                # Substract Image Vector from every correct classifier score vector
                dW[:, y[i]] -= X[i]
                # Add Image vector for every incorrect classifier score vector
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Divide all over training examples
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Scores will be of shape 500X1
    scores = np.dot(X, W)
    # Correct Class Scores after adding new axis will be of shape 500X1
    correct_class_scores = scores[np.arange(num_train), y][:, np.newaxis]
    # Substracting Correct Class Scores through Broadcasting
    # Keeping correct class loss in max function for now
    margin = np.maximum(0, scores - correct_class_scores + delta)
    # Making correct class loss to 0
    margin[np.arange(num_train), y] = 0
    # toClipboardForExcel(margin)

    # Summing Up Over All Loss
    loss = np.sum(margin)

    loss /= num_train

    # Adding Regularization Loss
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # This mask can flag the examples in which their margin is greater than 0
    count = np.zeros(margin.shape)
    count[margin > 0] = 1

    no_of_incorrect_predictions = np.sum(count, axis=1)
    count[np.arange(num_train), y] = -no_of_incorrect_predictions

    dW = X.T.dot(count)

    # Divide the gradient all over the number of training examples
    dW /= num_train

    # Regularize
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
