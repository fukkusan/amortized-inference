import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import pandas as pd



def Input():
    # Input data
    N = 1000  	# number of data points
    D = 2     	# dimensionality of data
    alpha = 3.0
    beta = 0.3


    x = nprand.multivariate_normal([alpha, alpha], [[beta, 0.0], [0.0, beta]], [N])
    
    x_mean = np.mean(x)
    
    
    data_frame = pd.DataFrame( index=[], columns=['element1', 'element2'] )		# null data frame
    for i in range(N):
        plt.scatter(x[i][0], x[i][1], color="r")
        series = pd.Series( [x[i][0], x[i][1]], index=data_frame.columns )
        data_frame = data_frame.append( series, ignore_index=True )
    plt.show()
    data_frame.to_csv("gauss_data.csv", index=False)


    return [x, x_mean]




