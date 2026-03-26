#--------------------- Machine Learning -------------------------
"""
Instructions

"""
print('----------------Machine learning - Exercise 5-----------------\n')
#-------------------- Data reading Function ---------------------
import reading
import numpy as np
from prediction import prediction, performance

filename="data/chess_white.csv"
X,y=reading.data(filename)
print('-------First 10 examples of the dataset---------')
print('            X             |  y')
print(np.hstack((X[:5,:],y[:5])))         #two features, one target
print('')

#-------------------- Plotting ------------------------
from plotting import viewData,viewCost,viewDecisionBoundary

viewData(X,y)

#-------------------- Cost Function -------------------
from logistic import costFunction, gradientDescent

m=X.shape[0]
X0=np.ones((m,1))           #Para ingresar b en w, X0=1
X=np.hstack((X0,X))         #Crear la matriz X=[x0,x1,x2]
n=X.shape[1]
w=np.zeros(n).reshape(n,1)
J=costFunction(X,y,w)
print(f'The initial cost is: {J[0].item():.4f}\n')

#-------------------- Gradient Descent -----------------

w=np.array([10,0,0]).reshape(n,1)
alpha=0.000003
n_iter=1000
J_hist,w=gradientDescent(X,y,w,alpha,n_iter)
print('----------------- w Values --------------------')
print('    wo     |      w1      |       w2')
print(f' {w[0].item():.4f}       {w[1].item():.4f}          {w[2].item():.4f}')
print('-----------------------------------------------\n')
viewCost(J_hist,alpha)
print(f'Final Cost after gradient descent execution:\n{J_hist[-1].item():.4f}\n')

#--------------------- Desicion Boundary ---------------
viewDecisionBoundary(X,y,w)

#--------------------- Prediction ----------------------
p = performance(w, X)
print(f'Train Accuracy: {(np.mean(p == y) * 100):.2f}%')

elo1 = 1800
elo2 = 1780
p = prediction(w,elo1,elo2)
print(f'Prediction for a game between a player with ELO {elo1} ' +
      f'and another with ELO {elo2}:\n{"white player wins" if p else "black player wins"}')

