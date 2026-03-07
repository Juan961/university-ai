import numpy as np

def normalEqu(X,y):
    m=X.shape[0]
    X0=np.ones((m,1))           #Para ingresar b en w, X0=1
    X=np.hstack((X0,X))         #Crear la matriz X=[1,x1,x2,x3,x4,x5]
    w=0
    
    #==================== Your code here ==========================
    #Instruction: Compute the normal function to get w = [1,w1,w2,w3,w4,w5]
    w = np.dot( np.linalg.inv( np.dot( X.T, X ) ), np.dot( X.T, y ) )
    #==============================================================
    return w
