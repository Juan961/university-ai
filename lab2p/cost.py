import numpy as np
import time

def CostFcn3D(X,y):
    #It creates a Mesh to structure the 3D graph
    w_range = np.array([-300,300])
    b_range = np.array([-200, 200])
    b_space  = np.linspace(*b_range, 200)       
    w_space  = np.linspace(*w_range, 200)

    tmp_b,tmp_w = np.meshgrid(b_space,w_space)
    J_hist=np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            J_hist[i,j] = costFcn(X, y, tmp_w[i][j], tmp_b[i][j] )
            if J_hist[i,j] == 0: J_hist[i,j] = 1e-6
    return J_hist,tmp_w,tmp_b

def costFcn(X,y,w,b):               #Function to evaluate the cost of a function
    #Instruction: create a function to evaluate the value of the cost for
    #             the whole dataset (X,y)
    #             1. build your hypothesis
    #             2. Calcule the Cost J for the evaluated hypothesis                               

    #==================== Write your code here ==========================
    y_pred = w * X + b
    J = np.sum( (y_pred - y) ** 2 ) / ( 2 * len(X))
    #==================================================================== 
    return J

def CostFcn2D(X,y,w=-500,b=100):    #Function to create a 2D graph with fixed b 
    iter=1000                       #Number of iterations to get to the global minimun
    J_hist=np.zeros((iter,2))       #Array to store the historical values of the cost               

    #Instruction: 1.create an iteration to evaluate the cost Function for
    #               every w with a fixed b.
    #             2.store each evaluated value of J in J_hist(first column)
    #             3.store each evaluated value of J in J_hist(second column)
    #               
    #                               'J_hist' => [J, w]

    #==================== Write your code here ==========================
    w_new = w

    for i in range(iter):
        w_new += 1
        J = costFcn(X, y, w_new, b)
        J_hist[i] = [J, w_new]
    #==================================================================== 
    return J,J_hist
