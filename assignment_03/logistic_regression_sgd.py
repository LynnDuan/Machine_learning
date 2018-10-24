#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
from random import randrange


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001
eps = 0.0001
eps1 = 0.0000001

# Step size for gradient descent.
eta = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]

# Error values over all iterations.
e_all = [ [] for x in range(len(eta))]
sgd_x = np.arange(len(X))
np.random.shuffle(sgd_x)

for i in range(len(eta)):
  w = np.array([0.1, 0, 0]) # Initialize w.

  for iter in range (0,max_iter):
    # Compute output using current w on all data X.
    y = sps.expit(np.dot(X,w))
    
    # e is the error, negative log-likelihood (Eqn 4.90)
    # e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
    e = -np.mean(np.multiply(t,np.log(y + eps1)) + np.multiply((1-t),np.log(1-y+eps1)+eps))
    for s in sgd_x:
      # Gradient of the error, using Eqn 4.91
      grad_e = np.multiply((y[s] - t[s]), X[s].T)
      w = w - eta[i]*grad_e
      y = sps.expit(np.dot(X,w))

    # Add this error to the end of error vector.
    e_all[i].append(e)

    # Stop iterating if error doesn't change more than tol.
    if iter>0:
      if np.absolute(e-e_all[i][iter-1]) < tol:
        break
colors = ['b','g','r','c','m']
plt.figure()
# Plot error over iterations
for j in range(len(e_all)):
  plt.plot([i for i in range(len(e_all[j]))],e_all[j],color = colors[j],label = "step size = "+str(eta[j]))
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.show()

