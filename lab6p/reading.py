import numpy as np

def data(filename):
    #================= You can delete this code ===================
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2]
    y = data[:, 2].reshape(-1, 1)
    #==============================================================

    #==================== Your code here ==========================
    #Instruction: use np.loadtxt to load the data: 
    #             3 columns [first_test, second_test, accepted(1)/Rejected(0)]
    
    #             then divide the dataset as follows:
    #             'X'=[first_test, second_test]
    #             'y'=[accepted(1)/Rejected(0)]
        
    #==============================================================   
    print('----------------Dimension------------------')
    print(f'Size X: {X.shape}, type:{type(X)}')
    print(f'Size y: {y.shape}, type:{type(y)}')
    print('-----------------------------------------\n')
    return X, y
