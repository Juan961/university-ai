import numpy as np
from sigmoid import sigmoid

def costFunction(X,y,w,lambd):
    m=X.shape[0]
    #================= You can delete this code ===================
    J = 1 / m * (-y.T @ np.log(sigmoid(X @ w)) - (1 - y).T @ np.log(1 - sigmoid(X @ w))) + lambd / (2 * m) * np.sum(np.square(w[1:]))
    h0 = sigmoid(X @ w)
    #==============================================================
    #==================== Your code here ==========================
    #Instruction: implement the cost function for a logistic regression
    #             1. compute gz= (Transpose(w)*X)
    #             2. compute the sigmoid function for gz
    #             3. compute J for logistic regression with REGULARIZATION
        
    #============================================================== 
    return J,h0

def gradientDescent(X,y,w,alpha,n_iter,lambd):            
    J_hist=[]
    m,n=X.shape
    grad=np.zeros((n))
    #==================== Your code here ==========================
    #Instruction: Implement the gradient descent algorithm to find
    #             the optimal values for w.
    #             this algorithm is similar to the one implemented
    #             for binary classification in the previous lab
    #             perhaps it should add the regularization term.

    #             Remember that the intercept term won't be regularized
    #==============================================================  
    for i in range(n_iter):
        J,h0=costFunction(X,y,w,lambd)
        J_hist.append(np.squeeze(J).item())
        grad = (1 / m) * (X.T @ (h0 - y)) + (lambd / m) * np.r_[[[0]], w[1:]]
        w = w - alpha * grad.reshape(-1, 1)
    return J_hist,w
