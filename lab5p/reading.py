import numpy as np

def data(filename):
    #================= You can delete this code ===================
    data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=np.object_)

    win_condition1 = data[:, 3] == "1-0"
    win_condition2 = data[:, 3] == "0-1"

    win_condition = win_condition1 | win_condition2

    clean_data = data[win_condition]

    elow = clean_data[:, 4].astype(np.float64)
    elob = clean_data[:, 5].astype(np.float64)

    difference = elow - elob
    average =  (elow + elob) / 2

    X_features = np.column_stack((difference, average))

    mask = clean_data[:, 3] == "1-0"

    y = mask.reshape(mask.shape[0], 1).astype(int)
    #==============================================================

    #==================== Your code here ==========================
    #Instruction: use np.loadtxt to load the data: 
    #             3 columns [Grades_exam1, Grades_exam2, Admitted(1)/NotAdmitted(0)]
    
    #             then divide the dataset as follows:
    #             'X'=[Grades_exam1, Grades_exam2]
    #             'y'=[Admitted(1)/NotAdmitted(0)]
        
    #==============================================================  
    print('----------------Dimension------------------')
    print(f'Size X: {X_features.shape}, type:{type(X_features)}')
    print(f'Size y: {y.shape}, type:{type(y)}')
    print('-----------------------------------------\n')
    return X_features, y