import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
K = 2
Step = 1000


# read csv
#dataset_lambda_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_lambda_pi.csv")
#dataset_lambda_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/res_lambda_pi_K2.csv")
dataset_lambda_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S100epoch1000sample100/1000stepOK2/res_lambda_pi_K2.csv")

lambda_pi_cl1_r = []
lambda_pi_cl1_r.append( dataset_lambda_pi['lambda_pi_class1'] )
#print(lambda_pi_cl1_r)

lambda_pi_cl2_r = []
lambda_pi_cl2_r.append( dataset_lambda_pi['lambda_pi_class2'] )
#print(lambda_pi_cl2_r)


lambda_pi_cl1 = []
for step in range(Step):
    lambda_pi_cl1.append( lambda_pi_cl1_r[0][step] )
print(lambda_pi_cl1)

lambda_pi_cl2 = []
for step in range(Step):
    lambda_pi_cl2.append( lambda_pi_cl2_r[0][step] )
print(lambda_pi_cl2)



# plot
fig = plt.figure()
ax = plt.axes()
ax.set_xlim(0,Step)
for epoch in range(Step):
    plt.subplot(K, 1, 1)
    plt.scatter( epoch, lambda_pi_cl1[epoch], color='b' )
    plt.ylabel("param class1")
    plt.subplot(K, 1, 2)
    plt.scatter( epoch, lambda_pi_cl2[epoch], color='r' )
    plt.ylabel("param class2")
    
    
    plt.suptitle("dirichlet parameter")
    
    #plt.pause(0.2)
plt.show()




