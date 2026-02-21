import matplotlib.pyplot as plt

def dataView(price, quantity):
    #==================== Your code here ==========================
    #Instruction: plot the price and quantity in relation with the 
    #             StoreId feature, you can create a subplot(1,2),
    #             then put the data as marks with its respectively
    #             xlabel, ylable, title and legend.
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(price[:,0], price[:,1], c="b", marker="x", label='Price')
    ax1.set_title("Price Graph")
    ax1.set_xlabel("StoreId")
    ax1.set_ylabel("Unit Price")
    ax1.legend()
    ax2.scatter(quantity[:,0], quantity[:,1], c="r", marker="x", label='Quantity')
    ax2.set_title("Quantity Graph")
    ax2.set_xlabel("StoreId")
    ax2.set_ylabel("Quantity")
    ax2.legend()

    plt.show()
    #==============================================================    
