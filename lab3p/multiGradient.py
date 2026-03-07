import numpy as np

def cost(h0,y,m):
    J=0
    #==================== Your code here ==========================
    #Instruction: Implement a vectorized cost function
    #             h0 => hypothesis
    #             y => Target
    #             m => total of samples 
    J = np.sum( (h0 - y) ** 2 ) / (2 * m)
    #============================================================== 
    return J

def gradientDescent(X,y,w,alpha,n_iter,m):
    features=X.shape[1]
    temp_w=np.zeros((features))
    J_hist=np.zeros((n_iter))

    for i in range(n_iter):
        #--------------- You can delete this code-lines ----------------
        err = np.dot(X, w) - y
        gradient = (X.T @ err) / m
        w = w - alpha * gradient
        h0 = np.dot(X, w)
        J_hist[i] = cost(h0, y, m)
        #---------------------------------------------------------------
        #==================== Your code here ==========================
        #Instruction: Implement the gradient descent algorithm for a
        #             multivariate function.
        #
        #Note: Remember that your code would run even faster if you 
        #      vectorize your algorithm
        #
        #      w: is a vector that returns the values for b,w1,w2,w3,w4,w5.
        #      J_hist: is a vector that returns the historical values of
        #              the cost function
        #
        #============================================================== 
    return w,J_hist

def multiGradient(X,y,w,alpha,n_iter):
    J_hist=[]                                   #To store the cost's historical values
    m = X.shape[0]                              #To get the amount of samples
    X0=np.ones((m,1))                           #To create a column of 1 to get the intercept unchanged
    X=np.hstack((X0,X))                         #Adding the column of 1 to X .... X=[1,x1,x2,x3,x4,x5]
    n = X.shape[1]                              #To get the number of features, including the intercept
    w_end=np.zeros((n,3))                       #Create a matrix to store the weights for the 3 different learning rates
    
    for i in range(3):                          #There are 3 different tests: for alpha =>1,0.1,0.01
        w, J=gradientDescent(X,y,w,alpha[i],n_iter,m)
        J_hist.append(J)
        w_end[:,i]=w.flatten()
        w=np.zeros((n)).reshape(n,1)            #Crear la matriz donde guardar w=[b,w1,w2,w3,w4,w5]
    return w_end,J_hist