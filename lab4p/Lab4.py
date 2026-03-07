#--------------------- Machine Learning -------------------------
"""
Instructions


"""
print('----------------Machine learning - Exercise 3-----------------\n')
#-------------------- Data reading Function ---------------------
import reading
import numpy as np

filename="dataset//diamonds.csv"
X,y=reading.data(filename)
print('First 10 examples of the dataset')
print(X[:10,:])
print(y[:10])
print(np.hstack((X[:10,:],y[:10])))
input('Press Enter to continue.................')

#------------------- Normal Equation --------------------------
from n_Equ import normalEqu

w = normalEqu(X,y)

#---------------------------- Prediction ------------------------------
#Get the prediction of a diamond with the following data:
#carat=0.76, depth=61.8, x=5.82, y=5.86, z=3.61

prediction = np.array([0.76,61.8,5.82,5.86,3.61])

#==================== Your code here ==========================
#Instruction: Use the values of w that you compute.
price = w[0, 0] + np.dot(w[1:].flatten(), prediction)
#Note: Remember not to scale the features of prediction too.
#============================================================== 
print(f'\nEstimated price: ${price:.2f}')
