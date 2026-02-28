import matplotlib.pyplot as plt
import numpy as np
from cost import CostFcn3D
from matplotlib.ticker import MaxNLocator

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; dlred='#A00000'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple, dlred]

def viewData(X,y,view=False):
    fig, ax = plt.subplots(figsize=(7,5))  # Create a figure containing a single axes.
    ax.scatter(X, y,marker='x', c='r', label="values")  # Plot some data on the axes.
    ax.set_xlabel('CO2 emission')
    ax.set_ylabel('Fuel cosumption')
    ax.set_title('Scatter plot - Training data')
    ax.legend()
    #---------------- To display the graph -------------------
    if view==False:
        plt.show() 
    else:
        return fig,ax

def linearRegression(X,y,w,b):
    fig,ax=viewData(X,y,True)
    #Instruction: Evaluate your hypothesis with the final values
    #             w and b that where found with the gradient descent
    #             algorithm. The use ax.plot(__,__) with the
    #             corresponding values of the X_axis and y_axis to 
    #             plot your hypothesis.
    #==================== Your code here ==========================
    y = w * X + b
    ax.plot(X, y)

    #==============================================================
    plt.show() 

def inbounds(a,b,xlim,ylim):
    xlow,xhigh = xlim
    ylow,yhigh = ylim
    ax, ay = a
    bx, by = b
    if (ax > xlow and ax < xhigh) and (bx > xlow and bx < xhigh) \
        and (ay > ylow and ay < yhigh) and (by > ylow and by < yhigh):
        return True
    return False

def contourWgrad(X, y, hist, ax, w_range=[-100, 500, 5], b_range=[-500, 500, 5],
                contours=[0.1, 50, 1000, 5000, 10000, 25000, 50000],
                resolution=5, w_final=0, b_final=0, J_final = 0, step=10):
    
    J_hist,tmp_w,tmp_b = CostFcn3D(X,y)

    CS = ax.contour(tmp_w, tmp_b, np.log(J_hist),levels=12, linewidths=2, alpha=0.7,colors=dlcolors)
    ax.set_title('Cost(w,b)')
    ax.set_xlabel('w', fontsize=10)
    ax.set_ylabel('b', fontsize=10)
    ax.scatter(w_final, b_final, color='blue', marker='X', s=100, label="Final Point")

    ax.hlines(b_final, ax.get_xlim()[0],w_final, lw=2, color=dlpurple, ls='dotted')
    ax.vlines(w_final, ax.get_ylim()[0],b_final, lw=2, color=dlpurple, ls='dotted')

    base = hist[0]
    for point in hist[0::step]:
        edist = np.sqrt((base[0] - point[0])**2 + (base[1] - point[1])**2)
        if edist > resolution or np.array_equal(point, hist[-1]):
            if inbounds(point,base, ax.get_xlim(),ax.get_ylim()):
                plt.annotate('', xy=point, xytext=base,xycoords='data',
                         arrowprops={'arrowstyle': '->', 'color': 'b', 'lw': 3},
                         va='center', ha='center')
            base=point


def soupBowl(w_space,b_space, J_hist):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_surface(w_space, b_space, J_hist,  alpha=0.3, color=dlblue)
    ax.xaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(2))

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel('\ncost', fontsize=16)
    plt.title('Cost vs (b, w)')
    # Customize the view angle
    ax.view_init(elev=20., azim=-65)
    ax.plot(w_space.ravel(), b_space.ravel(), J_hist.ravel(),c=dlmagenta)
    plt.show()
    return fig,ax


def viewCost(J_hist):
    J = J_hist[:,0]
    w = J_hist[:,1]
    plt.figure(figsize=(12, 4))         # Define the size to plot
    plt.plot(w,J, label="Costo J")      # Plot J in function of w
    plt.xlabel('w')                     # label x
    plt.ylabel('Costo J')               # label y
    plt.title('Evolución del costo J')  # Graph title
    plt.legend()                        # legend
    plt.show()                          # plotting data