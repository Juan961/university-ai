import matplotlib.pyplot as plt
import numpy as np
from sigmoid import sigmoid
from featuremult import mapfeature

def plotData(X,y,ax):
    #==================== Your code here ==========================
    #Instruction: classify the samples whether y is equal to 0 or 1
    #             if y==1 then use a marker (eg. +)
    #             if y==0 then use a marker (eg. o) 
    
    #             then plot both samples in the same figure with 
    #             scatters function.
    #             (eg. samples "+" in red color) 
    #             (eg. samples "o" in blue color)
        
    #============================================================== 
    pos = y.flatten() == 1
    neg = y.flatten() == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker='+', color='r', label='Accepted')
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', color='b', label='Rejected')
    ax.set_xlabel('First test')
    ax.set_ylabel('Second test')
    ax.set_title(f'Training data')
    ax.legend()
    return ax
    

def viewData(X,y):
    fig, ax = plt.subplots(figsize=(7,5))
    plotData(X,y,ax)
    plt.show()

def viewCost(J,alpha):
    fig, ax = plt.subplots()

    ax.plot(J,color='b',label=f'alpha={alpha}')
    ax.set_xlabel('# of iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.show()

def viewDecisionBoundary(X,y,w,lambd):
    fig, ax = plt.subplots(figsize=(7,5))
    plotData(X[:, 1:3], y,ax)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    
    for i in range(len(u)):
            for j in range(len(v)):
                X=np.array([[u[i],v[j]]])
                z[i,j] = sigmoid(np.dot(mapfeature(X,6), w)).item()            
    z = z.T
    
    contorno=plt.contour(u,v,z, levels = [0.5], colors="g")
    ax.plot([], [], color="g", label=f"lambda={lambd}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()
