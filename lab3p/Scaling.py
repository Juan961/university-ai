import numpy as np

def FeatureScaling(X):
    #--------------- You can delete this code-lines ----------------
    mu=0
    sigma=0
    X_norm=np.zeros((3))
    #--------------------------------------------------------------
    

    #==================== Your code here ==========================
    #Instruction: Implement feature scaling for every sample in the 
    #             dataset, you should use z-score normalization.
    #             mu: mean value
    #             sigma: standart deviation
    #             X_norm: feature scaling matrix
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    #============================================================== 

    return mu,sigma,X_norm