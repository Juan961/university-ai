#--------------------- Machine Learning -------------------------
"""
Instructions
Read the pdf document "Lab2.py" where you would find the task that you
must complete in this exercise
"""
print('----------------Machine learning - Exercise 2-----------------\n')
#-------------------- Data reading Function ---------------------
import reading

filename="dataset//data.txt"
X,y=reading.data(filename)
print(f'Tamaño X: {X.shape}')
print(f'Tamaño y: {y.shape}\n')

#--------------------Plotting Data Fuction ----------------------
import plotting as p

p.viewData(X,y)

#-------------------- Cost Function -----------------------------
import cost

J=cost.costFcn(X,y,20,100)
print('====================================================================\n',
     f'With w,b=[20,100] => Cost computed = {J}\n',
     '--------------------------------------------------------------------\n',
     'Expected aprox cost value => 4822.13\n',
     '====================================================================\n')
input('\nOprima enter para continuar.................')
#-------------------- Cost Function Plot ------------------------
from plotting import soupBowl

J,J_hist=cost.CostFcn2D(X,y)
print('====================================================================\n',
      f'With w,b=[-200,100] => Cost computed = {J_hist[300,0]}\n',
      '--------------------------------------------------------------------\n',
      'Expected aprox cost value => 5205782.45\n',
      '====================================================================\n')

p.viewCost(J_hist)
input('\nOprima enter para continuar.................')

#----------------------- SoupBowl Plot ------------------------
#You do not have to code nothing in this part
J_hist,w_space,b_space = cost.CostFcn3D(X,y)
soupBowl(w_space,b_space,J_hist)

#--------------- Gradient Descent Algorithm -------------------
from gradient import gradientDescent
temp_b=100
temp_w=20
param=[temp_b,temp_w]; alpha=0.008; n_iters=3000
print('Computing Gradient descent...')
J_hist,p_hist,param=gradientDescent(X,y,param,alpha,n_iters)
print('====================================================================\n',
      f'parameters found by gradient descent:\n b={param[0]} & w={param[1]}\n',
      '--------------------------------------------------------------------\n',
      f'Expected values for w and b\n b=[38.6957923] & w=[17.6383657]\n',
      '====================================================================\n')

#-------------------- Plotting hypothesis line ----------------------
b=param[0];w=param[1]
p.linearRegression(X,y,w,b)

# #-----------------------Visualizing J---------------------------
import matplotlib.pyplot as plt
from plotting import contourWgrad, soupBowl

J = cost.costFcn(X,y,w,b)               #get the cost of the final values of w and b
fig, ax = plt.subplots(1,1, figsize=(10, 6))
contourWgrad(X, y, p_hist, ax, w_range=[-0.5, 3, 0.05], b_range=[-20, 10, 0.05],
            contours=[1,5,10,20],resolution=0.5,w_final=w, b_final=b, J_final=J)
plt.show()

#----------------------- Predicted values -------------------------
from prediction import evaluationHypothesis
evaluationHypothesis(w,b)