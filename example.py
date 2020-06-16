# -*- coding: utf -*-
"""Test script.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@ieee.org
License: 3-clause BSD
Last modified: 16th June 2020
"""

import matplotlib.pyplot as plt

import numpy as np

from reg import polyfit, polynomial

import seaborn as sns

"""Load example dataset."""
data = np.load('data.npy', allow_pickle=True).flat[0]
x, y = data['x'], data['y']

"""Do regularized polynomial regression with different regularization
parameters."""
max_degree = 5
norm = 'L1' # must be either 'L1' or 'L2'
lambdas = [0, 0.05, 0.10]
n_lambdas = len(lambdas)

c = np.zeros([n_lambdas, max_degree+1], dtype='float')

for i, lambda_ in enumerate(lambdas):
    coefs = polyfit(x, y, lambda_, norm=norm, degree=max_degree)
    c[i, :] = coefs

"""Plot the results."""
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(x, y, '.', markersize=8)

for i in np.arange(0, n_lambdas):
    z = np.linspace(np.min(x), np.max(x), 1000) # smooth fitted curve
    p = polynomial(theta=c[i], x=z)
    ax.plot(z, p, '-')

ax.set_xlabel('Independent variable', fontsize=12)
ax.set_ylabel('Dependent variable', fontsize=12)
ax.set_title('Regularized polynomial regression', fontsize=14)

sns.despine(top=True, right=True)
fig.tight_layout()

plt.show()
