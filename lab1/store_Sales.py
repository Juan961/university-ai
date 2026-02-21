import numpy as np

def stores(data):
    #==================== Your code here ==========================
    #Instruction: from the whole dataset return just the following 
    #             columns ['SalesId','UnitPrice','Quantity']
    stores=data[:, [0,4,5]].astype(float)
    #==============================================================    
    return stores

def info(data):
    #==================== Your code here ==========================
    #Instruction: from the entry data, in price insert all the 
    #             elments from columns [0 and 1], then for quantity
    #             insert all the elements from columns [0 and 2] 
    price=data[:, [0, 1]].astype(float)
    quantity=data[:, [0, 2]].astype(int)
    #==============================================================    
    return price,quantity
