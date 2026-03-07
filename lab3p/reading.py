import numpy as np

def data(filename):
    #dataSet Headers => ['""', '"carat"', '"cut"', '"color"', '"clarity"', '"depth"', '"table"',
    #                     '"price"', '"x"', '"y"', '"z"']

    #Instruction: use np.loadtxt to get the data to store its individual values as follows:
    #             'X' => ['carat','depth','x','y','z'] 
    #             'Y' => ['price']
    #==================== Write your code here ==========================
    #The code in this area can be deleted.
    dataset = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[1, 5, 7, 8, 9, 10])
    X = dataset[:, [0, 1, 3, 4, 5]]
    y = dataset[:, 2:3]
    #==================================================================== 
    print(f'Tamaño X: {X.shape}, tipo:{type(X)}')
    print(f'Tamaño Y: {y.shape}, tipo:{type(y)}')
    return X.astype(float),y.astype(float)
