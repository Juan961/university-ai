
def evaluationHypothesis(w,b):
    #Instruction: Use the values gradient values for w and b to 
    #             generate a prediction:
    #             'Fuel Consumption of 25 liters' => 'CO2 emissions => ?'
    #             'Fuel Consumption of 32 liters' => 'CO2 emissions => ?'
    #==================== Your code here ============================
    print( f"Fuel Consumption of 25 liters => CO2 emissions => { w * 25 + b }" )
    print( f"Fuel Consumption of 32 liters => CO2 emissions => { w * 32 + b }" )
    #================================================================
