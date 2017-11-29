import numpy as np
import numpy.random as nprand


def Input():
    # Input data
    N = 500  	# number of data points
    D = 1     	# dimensionality of data
    alpha = 0.0
    beta = np.sqrt(0.1)
    gamma = 0.1
    rho = 0.1

    sample_x1 = nprand.normal(alpha, beta, int(float(N)*0.7))		# Mixture ratio pi_1=0.7
    sample_x2 = nprand.normal(alpha+3.0, beta, int(float(N)*0.2))	# Mixture ratio pi_2=0.2
    sample_x3 = nprand.normal(alpha-3.0, beta, int(float(N)*0.1))	# Mixture ratio pi_3=0.1
    x = np.reshape( np.concatenate( [np.concatenate([sample_x1, sample_x2]), sample_x3] ), (N, D) )

    x_mean = np.mean(x)


    return [x, x_mean]



