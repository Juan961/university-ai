import numpy as np
import sigmoid

def prediction(X,w):

    m,n=X.shape
    p = np.zeros(m)

    #==================== Your code here ==========================
    #Instruction: Use your found model to test the performance over X
    #             With [w] compute the estimated output "y", then 
    #             create a vector p to storage the output whether your 
    #             model predicts 1 or 0.
    #             finally p is returned to compare the estimated
    #             output with the model target.
    #             Same implementation just like your previous lab 
    p = (sigmoid.sigmoid(X @ w) >= 0.5).astype(int)
    #==============================================================
    return p

