import numpy as np
import numpy.random as rand



D=2
K=3
epsilon = 0.0001
nu = rand.uniform(low=D-1+epsilon, high=30, size=K)
print(nu)


