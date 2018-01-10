import numpy as np
import numpy.random as rand
import scipy.stats as static



Lambda = static.wishart.rvs(df=20.0, scale=[[0.5, 0.0], [0.0, 0.5]])
eig = np.linalg.eig(Lambda)
print(Lambda)
print(eig[0])

