# reading.py is a function to get the data from an external file.csv
import numpy as np

def dataReading(filename): 
    #==================== Your code here ==========================
    #Instruction: use np.loadtxt to get the data as str. Besides
    #             use the instruction split to create a list of 
    #             elements delimited by ','
    dataset=np.loadtxt(filename, dtype=str, delimiter=",")  
    #==============================================================    
    return dataset
