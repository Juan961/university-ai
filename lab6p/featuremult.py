import numpy as np

def mapfeature(X,grade):
    m=X.shape[0]
    out=np.ones(m).reshape(m,1)         #already creates the first term for the intercept
    #==================== Your code here =====================================
    #Instruction: implement an algorithm to create more features from x1 & x2.
    #             the idea is to map the features into polinomial terms of x1 
    #             and x2 to sixth power, as follows.

    #                                 (1)
    #                           (x1^1)   (x2^1)
    #                      (x1^2)  (x1^1*x2^1)  (x2^2)
    #                   ...   ...   ...   ...   ...   ...
    #            (x1^4) (x1^3*x2^1) (x1^2*x2^2) (x1^1 x2^3) (x2^4)
    #           ...   ...   ...   ...   ...   ...   ...   ...   ...

    #              Note: to test your implementation, you can execute this file
    #                    directly. if your code is correct the terminal shows
    #                    "You passed the test!"
    #==========================================================================
    for i in range(1,grade+1):
        for j in range(i+1):
            out=np.hstack((out,(X[:,0]**(i-j)*X[:,1]**j).reshape(m,1)))
    return out


#------------------------------------------ Unitary Test ------------------------------------------------
def test(target):
    np.random.seed(1)
    X=np.array([[4,5],[3,2]])
    expected_output = target(X,6)
    output=np.array([[1.0000e+00, 4.0000e+00, 5.0000e+00, 1.6000e+01, 2.0000e+01, 2.5000e+01,
                                          6.4000e+01, 8.0000e+01, 1.0000e+02, 1.2500e+02, 2.5600e+02, 3.2000e+02,
                                          4.0000e+02, 5.0000e+02, 6.2500e+02, 1.0240e+03, 1.2800e+03, 1.6000e+03,
                                          2.0000e+03, 2.5000e+03, 3.1250e+03, 4.0960e+03, 5.1200e+03, 6.4000e+03,
                                          8.0000e+03, 1.0000e+04, 1.2500e+04, 1.5625e+04],
                                          [1.0000e+00, 3.0000e+00, 2.0000e+00, 9.0000e+00, 6.0000e+00, 4.0000e+00,
                                           2.7000e+01, 1.8000e+01, 1.2000e+01, 8.0000e+00, 8.1000e+01, 5.4000e+01,
                                           3.6000e+01, 2.4000e+01, 1.6000e+01, 2.4300e+02, 1.6200e+02, 1.0800e+02,
                                           7.2000e+01, 4.8000e+01, 3.2000e+01, 7.2900e+02, 4.8600e+02, 3.2400e+02,
                                           2.1600e+02, 1.4400e+02, 9.6000e+01, 6.4000e+01]])
    
    assert np.allclose(expected_output, output ), f"Wrong output. Expected: {'\033[91mRevise su implementación, Respuesta erronea\033[97m'}"
    print('\033[92mYou have passed the test!\033[97m')

if __name__=="__main__":
    test(mapfeature)