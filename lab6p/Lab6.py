#--------------------- Machine Learning -------------------------
"""
Instructions
Execute Lab5.py script after you finish to code the exercises

"""
print('----------------Machine learning - Exercise 5-----------------\n')
#-------------------- Data reading Function ---------------------
import reading
import numpy as np

filename="dataset//data2.txt"
X,y=reading.data(filename)
print('-------First 10 examples of the dataset---------')
print('            X             |  y')
print(np.hstack((X[:10,:],y[:10])))     #two features, one target
print('')

#-------------------- Plotting ------------------------
from plotting import viewData, viewCost, viewDecisionBoundary
viewData(X,y)                           #Function to plot the data of X according to y

#--------------- Polinomial features multiplication -------------
from featuremult import mapfeature

X=mapfeature(X,6)

#-------------------- Cost Function ---------------------
from logistic import costFunction, gradientDescent

n=X.shape[1]
w=np.zeros(n).reshape(n,1)
lambd=1
J=costFunction(X,y,w,lambd)
print(f'First Cost after gradient descent execution:\n{J[0].item():.4f}\n')

#---------- Gradient Descent without Regularization-------------------
w=np.zeros((n,1))
#==================== Your code here ==========================
#Instruction: Select a proper value for alpha and n_iter.
#            you must change the values of alpha and n_iter
alpha=0.005
n_iter=100000
#==============================================================
J_hist,w=gradientDescent(X,y,w,alpha,n_iter,0)

print('----------------- w Values --------------------')
print('    wo     |      w1      |       w2')
print(f' {w[0].item():.4f}       {w[1].item():.4f}          {w[2].item():.4f}')
print('-----------------------------------------------\n')
viewCost(J_hist,alpha)
print(f'Final Cost after gradient descent execution:\n{J_hist[-1]:.4f}\n')

#-------------------- Gradient Descent -------------------
w=np.zeros((n,1))
lambd=0                 #Regularization term
J_hist,w=gradientDescent(X,y,w,alpha,n_iter,lambd)

#--------------------- Desicion Boundary ---------------
viewDecisionBoundary(X,y,w,lambd)

# ------------------------- Prediction --------------------------
from predict import prediction

p = prediction(X,w)
print(f'Train Accuracy: {(np.mean(p == y) * 100):.2f}%')
