import numpy as np

def sigmoid(z):
    z = np.clip( z, -500, 500 )     #prevents large values of z greater or lower than 500

    #==================== Your code here ==========================
    #Instruction: Implement the sigmoid function and return its value
    #             in the variable g.
        
    #==============================================================
    g = 1 / (1 + np.exp(-z))
    return g


if __name__=="__main__":
    #Try it: a large positive values sigmoid should be close to 1
    #        a large negative values sigmoid should be close to 0
    #        a value of zero should be 0.5
    z=int(input('Ingrese un valor: '))
    g=sigmoid(z)
    print(g)