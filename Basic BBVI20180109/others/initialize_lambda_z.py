import numpy as np
import numpy.random as rand



N = 10
K = 3
pi = rand.dirichlet(alpha=[0.2, 0.2, 0.2], size=N)
print(pi)

