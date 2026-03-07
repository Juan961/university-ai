#-------------------------- Machine Learning ---------------------------
"""
Instructions
Read the pdf document "Lab3.py" where you would find the task that you
must complete in this exercise

"""
print('----------------Machine learning - Exercise 3-----------------\n')
#-------------------------- Data reading Function ----------------------
import reading
import numpy as np

filename="dataset//diamonds1.csv"
X,y=reading.data(filename)
print('First 5 examples of the dataset')
print(np.hstack((X[:5,:],y[:5])))
input('Press enter to continue.................')

#-------------------------- Feature Scaling ----------------------------
from Scaling import FeatureScaling

mu,sigma,X_norm=FeatureScaling(X)
print('\nFirst 5 examples of the dataset')
print(np.hstack((X_norm[:5,:],y[:5])))
input('Press enter to continue.................')

#-------------------------- Gradient descent ---------------------------
from multiGradient import multiGradient
from plotting import viewCost

alpha=[0.1,0.01,0.001]           #Values to compute the gradient with 3 different learning rates
n_iter=400                      #Amount of iterations
m,n=X.shape
w=np.zeros((n+1)).reshape(n+1,1)
w,J_hist=multiGradient(X_norm,y,w,alpha,n_iter)
print(f'\nLos gradientes son:')
print('[[b],[w1],[w2],[w3],[w4],[w5]]\n')
print(f'    alpha={alpha[0]}      |   alpha={alpha[1]}   |   alpha={alpha[2]}')
print(w)

viewCost(J_hist,alpha)

#---------------------------- Prediction ------------------------------
#Get the prediction of a diamond with the following data:
#carat=0.76, depth=61.8, x=5.82, y=5.86, z=3.61

prediction=np.array([0.76,61.8,5.82,5.86,3.61])

#==================== Your code here ==========================
#Instruction: Use the values of alpha that you selected.
price = np.dot( w[1:, 1], (prediction - mu) / sigma ) + w[0, 1]
#Note: Remember to scale the features of prediction too.
#============================================================== 
print(f'\nEstimated price: ${price}')
