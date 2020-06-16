# -*- coding: utf-8 -*-
"""Functions for regularized regression and classification.

TODO: Choice of solver?
TODO: Automated testing ...

Author: Tuomas Puoliväli
Email: tuomas.puolivali@ieee.org
Last modified: 12th May 2020
License: Revised 3-clause BSD
"""

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize

def regularize(loss_function, theta, regularizer, lambda_):
    """Function for evaluating a regularized loss function.

    Input arguments:
    ================
    loss_function : function
        Loss function to be regularized.

    theta : ndarray
        Loss function parameters.

    lambda_ : float
        Regularization parameter.

    regularizer : function
        Regularizer.

    Output arguments:
    =================
    y : float
        Regularized loss function evaluated at theta.
    """
    y = loss_function(theta) + lambda_ * regularizer(theta)
    return y

def polynomial(theta, x):
    """Function for evaluating a polynomial function.

    Input arguments:
    ================
    theta : ndarray
        Coefficients of the polynomial.

    x : float
        Argument of the polynomial function.
    """
    n_terms = len(theta)
    y = 0
    for i in np.arange(n_terms-1, 0, -1):
        y += theta[i] * (x ** i)
    return y

def polyfit(x, y, lambda_, norm='L2', degree=5):
    """Function for regularized polynomial regression.

    Input arguments:
    ================
    x, y : ndarray [n_samples, ]
        The dependent and independent variables.

    lambda_ : float
        Regularization parameter.

    norm : str
        Regularizer, which must be either 'L1' or 'L2'.

    degree : int
        Order of the polynomial.

    Output arguments:
    =================
    coefs : ndarray [degree+1, ]
        Coefficients of the fitted polynomial.
    """

    """Setup the regularization function. NOTE: the convention is to not
    include the constant term in the regularizer."""
    if (norm == 'L1'):
        regularizer = lambda theta: np.sum(np.abs(theta[0:-1]))
    elif (norm == 'L2'):
        regularizer = lambda theta: np.sum(theta[0:-1]**2)
    else:
        print('ERROR: unrecognized norm %s' % norm)
        return

    """Setup the loss and regularized loss functions."""
    loss = lambda theta: np.sum((y - polynomial(theta, x)) ** 2)
    reg_loss = lambda theta: regularize(loss, theta,
                                        regularizer, lambda_)

    """Fit the model."""
    x0 = np.random.random(size=degree+1)
    result = minimize(reg_loss, x0, method='nelder-mead',
                      options={'disp': True, 'maxiter': 1e5})
    coefs = result.x
    return coefs
