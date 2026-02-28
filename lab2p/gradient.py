import numpy as np
from cost import costFcn

def gradientDescent(X,y,param,alpha,n_iters):
    m=X.shape[0]
    X=X.reshape(m,1)
    y=y.reshape(m,1)
    
    J_hist=np.zeros((n_iters,1))
    p_hist=np.zeros((n_iters,2))
    b=param[0];w=param[1]

    #Instruction: to calculate the gradient, remember that this process
    #             must be iterative, evaluating the values for the cost 
    #             function at every w and b.
    #
    #             'J_hist' => historical values for the cost function
    #             'p_hist' => historical values for w and b
    #             'param' => converge values for w and b
    #
    #---------------------- Code Structure -------------------------
    #for __________:
    #    Calculate the gradient
    #    Update the values for every w and b
    #    Store the J and gradient values (w,b) at every iteration
    #endfor
    #return J_hist,p_hist,param
        
    #==================== Your code here ==========================
    for i in range(n_iters):
        err = ( w * X + b - y)

        dw = np.sum( err * X ) / m
        db = np.sum( err ) / m

        w = w - alpha * dw
        b = b - alpha * db

        J = costFcn(X, y, w, b)

        J_hist[i] = J
        p_hist[i] = [w, b]

    #============================================================== 
    param=[b,w]             #Final gradient values for b and w
    
    return J_hist,p_hist,param