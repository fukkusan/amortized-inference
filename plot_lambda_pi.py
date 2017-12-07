import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
K = 3
Step = 2


# read csv
dataset_lambda_pi = pd.read_csv("res_lambda_pi.csv")

lambda_pi_cl1_r = []
lambda_pi_cl1_r.append( dataset_lambda_pi['lambda_pi_class1'] )
#print(lambda_pi_cl1_r)

lambda_pi_cl2_r = []
lambda_pi_cl2_r.append( dataset_lambda_pi['lambda_pi_class2'] )
#print(lambda_pi_cl2_r)

lambda_pi_cl3_r = []
lambda_pi_cl3_r.append( dataset_lambda_pi['lambda_pi_class3'] )
#print(lambda_pi_cl3_r)

lambda_pi_cl1 = []
for step in range(Step):
    lambda_pi_cl1.append( lambda_pi_cl1_r[0][step] )
print(lambda_pi_cl1)

lambda_pi_cl2 = []
for step in range(Step):
    lambda_pi_cl2.append( lambda_pi_cl2_r[0][step] )
print(lambda_pi_cl2)

lambda_pi_cl3 = []
for step in range(Step):
    lambda_pi_cl3.append( lambda_pi_cl3_r[0][step] )
print(lambda_pi_cl3)


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
    plt.subplot(K, 1, 3)
    plt.scatter( epoch, lambda_pi_cl3[epoch], color='g' )
    plt.xlabel("epoch")
    plt.ylabel("param class3")
    
    plt.suptitle("dirichlet parameter")
    
    plt.pause(2.2)




