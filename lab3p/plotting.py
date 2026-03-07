import matplotlib.pyplot as plt
import numpy as np
from multiGradient import cost

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; dlred='#A00000'

def viewCost(J,alpha):
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    colores=['r','g','b']

    for idx,J_sample in enumerate(J):
        axs[idx].plot(J_sample,color=colores[idx],label=f'alpha={alpha[idx]}')
        axs[idx].set_xlabel('# of iterations')
        axs[idx].set_ylabel('Loss')
        axs[idx].set_title(f'alpha {alpha[idx]}')
        axs[idx].legend()
    plt.tight_layout()
    plt.show()

