import numpy as np
from sigmoid import sigmoid

def costFunction(X,y,w):
    m=X.shape[0]
    #================= You can delete this code ===================
    J=0
    h0=0 
    #==============================================================
    #==================== Your code here ==========================
    #Instruction: implement the cost function for a logistic regression
    #             1. compute gz= (Transpose(w)*X)
    #             2. compute the sigmoid function for gz
    #             3. compute J for logistic regression
        
    #============================================================== 
    gz = np.dot(X, w)
    h0 = sigmoid(gz)
    J = (-1/m) * np.sum(y * np.log(h0) + (1 - y) * np.log(1 - h0))
    return J,h0

def gradientDescent(X,y,w,alpha,n_iter):            
    J_hist=[]
    m,n=X.shape
    grad=np.zeros((n))
    
    #==================== Your code here ==========================
    #Instruction: Implement the gradient descent algorithm to find
    #             the optimal values for w.
    #             this algorithm is similar to the one implemented
    #             for linear regression, perhaps its hypothesis 
    #             changes.
        
    #==============================================================
    for i in range(n_iter):
        gz = np.dot(X, w)
        h0 = sigmoid(gz)
        grad = (1/m) * np.dot(X.T, (h0 - y))
        w = w - alpha * grad
        J, _ = costFunction(X, y, w)
        J_hist.append(J)

    return J_hist,w
