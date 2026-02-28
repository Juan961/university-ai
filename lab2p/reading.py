import numpy as np

def data(filename):
    
    #Instruction: use np.loadtxt to get the data to store its individual
    #             values as follows:
    #             'dataset' => [Fuel_Consumption, CO2_emmission] comes from np.loadtxt
    #             'X' => (Fuel_Consumption) 
    #             'y' => (CO2_emmission)
    #==================== Write your code here ==========================
    #The code in this area can be deleted.
    dataset = np.loadtxt(filename, np.float32, delimiter=",")
    X = dataset[:,0]
    y = dataset[:,1]
    #==================================================================== 
    print(dataset[:3,:])
    return X, y