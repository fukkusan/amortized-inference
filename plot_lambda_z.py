import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
N = 10
K = 3
Step = 2


# read csv
dataset_lambda_z = pd.read_csv("res_lambda_z.csv")

lambda_z_cl1_r = []
lambda_z_cl1_r.append(dataset_lambda_z['lambda_z_class1'])
#print(lambda_z_cl1_r[0][10])

lambda_z_cl2_r = []
lambda_z_cl2_r.append(dataset_lambda_z['lambda_z_class2'])
#print(lambda_z_cl2_r)

lambda_z_cl3_r = []
lambda_z_cl3_r.append(dataset_lambda_z['lambda_z_class3'])
#print(lambda_z_cl3_r)


_lambda_z_cl1 = []
len_cl1_r = len(lambda_z_cl1_r[0])
for i in range(len_cl1_r):
    _lambda_z_cl1.append( lambda_z_cl1_r[0][i] )
#print(_lambda_z_cl1)

_lambda_z_cl2 = []
len_cl2_r = len(lambda_z_cl2_r[0])
for i in range(len_cl2_r):
    _lambda_z_cl2.append( lambda_z_cl2_r[0][i] )
#print(_lambda_z_cl2)

_lambda_z_cl3 = []
len_cl3_r = len(lambda_z_cl3_r[0])
for i in range(len_cl3_r):
    _lambda_z_cl3.append( lambda_z_cl3_r[0][i] )
#print(_lambda_z_cl3)


lambda_z_cl1 = []
start1 = 0
len_cl1 = len(_lambda_z_cl1)
while(start1 < len_cl1):
    lambda_z_cl1.append( _lambda_z_cl1[start1:start1+N] )
    start1 = start1 + N
#print(lambda_z_cl1[1][3])

lambda_z_cl2 = []
start2 = 0
len_cl2 = len(_lambda_z_cl2)
while(start2 < len_cl2):
    lambda_z_cl2.append( _lambda_z_cl2[start2:start2+N] )
    start2 = start2 + N
#print(lambda_z_cl2)

lambda_z_cl3 = []
start3 = 0
len_cl3 = len(_lambda_z_cl3)
while(start3 < len_cl3):
    lambda_z_cl3.append( _lambda_z_cl3[start3:start3+N] )
    start3 = start3 + N
#print(lambda_z_cl3)



for n in range(N):
    fig = plt.figure()
    ax = plt.axes()
    for epoch in range(Step):
        plt.subplot(K, 1, 1)
        plt.scatter(epoch, lambda_z_cl1[epoch][n], color='b')
        plt.ylabel("param class1")
        plt.xlim(0, Step)
        plt.subplot(K, 1, 2)
        plt.scatter(epoch, lambda_z_cl2[epoch][n], color='r')
        plt.ylabel("param class2")
        plt.xlim(0, Step)
        plt.subplot(K, 1, 3)
        plt.scatter(epoch, lambda_z_cl3[epoch][n], color='g')
        plt.ylabel("param class3")
        plt.xlabel("epoch")
        plt.xlim(0, Step)
        
        plt.suptitle("categorical parameter")
        
        plt.pause(1.2)
        








