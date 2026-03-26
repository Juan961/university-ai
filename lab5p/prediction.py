from sigmoid import sigmoid
import numpy as np

def prediction(w,Elo1,Elo2):
    p=0
    #==================== Your code here ==========================
    #Instruction: Create a function to predict the output for a game between 
    #             two players with ELO ratings Elo1 and Elo2.
    #             1. compute the features for the input data (Elo1 and Elo2) 
    #                as you did for the training data.
    #             2. compute gz= (Transpose(w)*X) where X is the feature vector 
    #                for the input data.
    #             3. compute the sigmoid function for gz to get the predicted 
    #                probability of the white player winning.
    #             4. if the predicted probability is greater than or equal to 
    #                0.5, predict that the white player wins (return 1), 
    #                otherwise predict that the black player wins (return 0).       
    #==============================================================
    difference = Elo1 - Elo2
    average = (Elo1 + Elo2) / 2
    gz = w[0] + w[1] * difference + w[2] * average
    predicted_probability = sigmoid(gz)
    p = int(predicted_probability.item() >= 0.5)
    return p


def performance(w, X):
    m,n=X.shape
    p = np.zeros(m)

    #==================== Your code here ==========================
    #Instruction: Use your found model to test the performance over X
    #             With [w] compute the estimated output "y", then 
    #             create a vector p to storage the output whether your 
    #             model predicts 1 or 0.
    #             finally p is returned to compare the estimated
    #             output with the model target. 
        
    #==============================================================
    gz = np.dot(X, w)
    predicted_probabilities = sigmoid(gz)
    p = (predicted_probabilities >= 0.5).astype(int)
    return p