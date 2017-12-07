import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
K = 3
Step = 2


# read csv
dataset_lambda_nu = pd.read_csv("res_lambda_nu.csv")

lambda_nu_cl1_r = []
lambda_nu_cl1_r.append( dataset_lambda_nu['lambda_nu_class1'] )
#print(lambda_nu_cl1_r)

lambda_nu_cl2_r = []
lambda_nu_cl2_r.append( dataset_lambda_nu['lambda_nu_class2'] )
#print(lambda_nu_cl2_r)

lambda_nu_cl3_r = []
lambda_nu_cl3_r.append( dataset_lambda_nu['lambda_nu_class3'] )
#print(lambda_nu_cl3_r)


lambda_nu_cl1 = []
for step in range(Step):
    lambda_nu_cl1.append( lambda_nu_cl1_r[0][step] )
#print(lambda_nu_cl1)

lambda_nu_cl2 = []
for step in range(Step):
    lambda_nu_cl2.append( lambda_nu_cl2_r[0][step] )
#print(lambda_nu_cl2)

lambda_nu_cl3 = []
for step in range(Step):
    lambda_nu_cl3.append( lambda_nu_cl3_r[0][step] )
#print(lambda_nu_cl3)


fig = plt.figure()
ax = plt.axes()
for epoch in range(Step):
    plt.subplot(K, 1, 1)
    plt.scatter(epoch, lambda_nu_cl1[epoch], color='b')
    plt.ylabel("dof class1")
    plt.xlim([0, Step])
    plt.subplot(K, 1, 2)
    plt.scatter(epoch, lambda_nu_cl2[epoch], color='r')
    plt.ylabel("dof class2")
    plt.xlim([0, Step])
    plt.subplot(K, 1, 3)
    plt.scatter(epoch, lambda_nu_cl3[epoch], color='g')
    plt.xlabel("epoch")
    plt.ylabel("dof class3")
    plt.xlim([0, Step])
    
    plt.suptitle("degree of freedom")
    
    plt.pause(1.2)





