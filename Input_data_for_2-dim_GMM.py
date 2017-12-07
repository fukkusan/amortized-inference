import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import pandas as pd



def Input():
    # Input data
    N = 10  	# number of data points
    D = 2     	# dimensionality of data
    alpha = 0.0
    beta = np.sqrt(0.1)
    gamma = 0.1

    sample_x1 = nprand.normal( alpha, beta, [int(float(N)*0.7), D] )		# Mixture ratio pi_1=0.7
    sample_x2 = nprand.normal( alpha+3.0, beta, [int(float(N)*0.2), D] )	# Mixture ratio pi_2=0.2
    sample_x3 = nprand.normal( alpha-3.0, beta, [int(float(N)*0.1), D] )	# Mixture ratio pi_3=0.1
    x = np.reshape( np.concatenate( [np.concatenate([sample_x1, sample_x2]), sample_x3] ), (N, D) )

    x_mean = np.mean(x)


    return [x, x_mean]



## Main (for debug)
#if __name__ == '__main__':
#    N = 10
#    x = Input()[0]
#    #print(x)
#    #print(x.shape)
#    data_frame = pd.DataFrame( index=[], columns=['element1', 'element2'] )		# null data frame
#    for i in range(N):
#        plt.scatter(x[i][0], x[i][1], color="r")
#        series = pd.Series( [x[i][0], x[i][1]], index=data_frame.columns )
#        data_frame = data_frame.append( series, ignore_index=True )
#    plt.show()
#    data_frame.to_csv("gmm_data.csv", index=False)


