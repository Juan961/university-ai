import matplotlib.pyplot as plt
import numpy as np

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
    ax.scatter(X[y[:,0]==1][:,0], X[y[:,0]==1][:,1], marker='+', color='r', label='white wins')
    ax.scatter(X[y[:,0]==0][:,0], X[y[:,0]==0][:,1], marker='o', color='b', label='black wins')
    ax.set_xlabel('ELO diff')
    ax.set_ylabel('ELO average')
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

def viewDecisionBoundary(X,y,w):
    fig, ax = plt.subplots(figsize=(7,5))
    #==================== Your code here ==========================
    #Instruction: Draw the decision boundary for your model by 
    #             specifying two points to drawa line.
    #             (x[1],y[1])----------------------(x[2],y[2])
    
    #Note: x_axis: get the min and max value in X samples.
    #      y_axis: if y=1 then z=0, that is, solve the equation of z
    #              for x1 or x2.
    #              ( z = 0 )=> x1*w1 + x2*w2 + b = 0 
    #      finally with ax.plot() include the points of x_axis & y_axis
    #      you can use linestyle='dashed' to draw a dashed line
        
    #==============================================================
    X_plot = X[:, 1:]
    plotData(X_plot, y, ax)

    min = np.min(X_plot[:, 0])
    max = np.max(X_plot[:, 0])
    x_axis = np.linspace(min, max, 100)

    y_axis = -(w[0] + w[1] * x_axis) / w[2]
    ax.plot(x_axis, y_axis, linestyle='dashed', color='g', label='decision boundary')
    ax.set_title('Decision Boundary')

    plt.show()