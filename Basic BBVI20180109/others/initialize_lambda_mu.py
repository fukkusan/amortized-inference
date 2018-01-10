import numpy as np
import numpy.random as rand


mu1 = rand.multivariate_normal(mean=[3.0, 0.0], cov=[[0.1, 0.0], [0.0, 0.1]])
mu2 = rand.multivariate_normal(mean=[0.0, 3.0], cov=[[0.1, 0.0], [0.0, 0.1]])
print(mu1)
print(mu2)



