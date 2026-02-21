#--------------------- Machine Learning -------------------------
"""
Instructions
Execute Lab1.py script after you finish to code the exercises 
"""
print('----------------Machine learning - Exercise 1-----------------\n')
#-------------------- Data reading Function ---------------------
import reading

filename="dataset//Sales.csv"
sales=reading.dataReading(filename) 
print('These are the first 3 rows of the dataset __Sales__:\n')
print(sales[:3],'\n')


filename="dataset//Inventory.csv"
inventory=reading.dataReading(filename)
print('These are the first 3 rows of the dataset __Inventory__:\n')
print(inventory[:3],'\n')

#-------------------- Data Manipulation -------------------------
import store_Sales as ss
import numpy as np

data=ss.stores(sales[1:])
data=np.array(data)
print('Printing the first 10 elements of the dataset (SalesId,UnitPrice,Quantity)...')
print(f'Dataset size: {data.shape}\n')
print(data[:10,:])

#-------------------- Data Specification ------------------------

data1,data2=ss.info(data)
print(f'\nUnit Price dataset ... \n{data1[:5,:]}\n')
print(f'Quantity dataset ... \n{data2[:5,:]}\n')

#---------------------- Plotting Data ---------------------------
import plotting as p

p.dataView(data1[:200,:],data2[:200,:])