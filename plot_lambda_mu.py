import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
K = 3
Step = 100



# read csv
dataset_lambda_mu = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_lambda_mu.csv")


lambda_mu_cl1_el1 = []
lambda_mu_cl1_el1.append( dataset_lambda_mu['lambda_mu_class1_element1'] )
#print(lambda_mu_cl1_el1)

lambda_mu_cl1_el2 = []
lambda_mu_cl1_el2.append( dataset_lambda_mu['lambda_mu_class1_element2'] )
#print(lambda_mu_cl1_el2)

lambda_mu_cl2_el1 = []
lambda_mu_cl2_el1.append( dataset_lambda_mu['lambda_mu_class2_element1'] )
#print(lambda_mu_cl2_el1)

lambda_mu_cl2_el2 = []
lambda_mu_cl2_el2.append( dataset_lambda_mu['lambda_mu_class2_element2'] )
#print(lambda_mu_cl2_el2)

lambda_mu_cl3_el1 = []
lambda_mu_cl3_el1.append( dataset_lambda_mu['lambda_mu_class3_element1'] )
#print(lambda_mu_cl3_el1)

lambda_mu_cl3_el2 = []
lambda_mu_cl3_el2.append( dataset_lambda_mu['lambda_mu_class3_element2'] )
#print(lambda_mu_cl3_el2)


# mean each classes
lambda_mu_cl1 = []
for step in range(Step):
    lambda_mu_cl1.append( [ lambda_mu_cl1_el1[0][step], lambda_mu_cl1_el2[0][step] ] )
lambda_mu_cl1_ndarray = np.array(lambda_mu_cl1)
#print(lambda_mu_cl1[1])

lambda_mu_cl2 = []
for step in range(Step):
    lambda_mu_cl2.append( [ lambda_mu_cl2_el1[0][step], lambda_mu_cl2_el2[0][step] ] )
lambda_mu_cl2_ndarray = np.array(lambda_mu_cl2)
#print(lambda_mu_cl2)

lambda_mu_cl3 = []
for step in range(Step):
    lambda_mu_cl3.append( [ lambda_mu_cl3_el1[0][step], lambda_mu_cl3_el2[0][step] ] )
lambda_mu_cl3_ndarray = np.array(lambda_mu_cl3)
#print(lambda_mu_cl3)


# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(Step):
    ax.cla()
    plt.suptitle("mean parameter(lambda_mu) of Gaussian")
    plt.subplot(K, 1, 1)
    if(epoch==0):
        plt.scatter(epoch, lambda_mu_cl1[epoch][0], color='cyan', label="element1")
        plt.scatter(epoch, lambda_mu_cl1[epoch][1], color='midnightblue', label="element2")
    else:
        plt.scatter(epoch, lambda_mu_cl1[epoch][0], color='cyan')
        plt.scatter(epoch, lambda_mu_cl1[epoch][1], color='midnightblue')
    #plt.quiver(0, 0, lambda_mu_cl1[epoch][0], lambda_mu_cl1[epoch][1])
    plt.xlim([0, Step])
    plt.ylabel("param class1")
    plt.legend()
    plt.subplot(K, 1, 2)
    if(epoch==0):
        plt.scatter(epoch, lambda_mu_cl2[epoch][0], color='tomato', label="element1")
        plt.scatter(epoch, lambda_mu_cl2[epoch][1], color='salmon', label="element2")
    else:
        plt.scatter(epoch, lambda_mu_cl2[epoch][0], color='tomato')
        plt.scatter(epoch, lambda_mu_cl2[epoch][1], color='salmon')
    plt.xlim([0, Step])
    plt.ylabel("param class2")
    plt.legend()
    plt.subplot(K, 1, 3)
    if(epoch==0):
        plt.scatter(epoch, lambda_mu_cl3[epoch][0], color='lime', label="element1")
        plt.scatter(epoch, lambda_mu_cl3[epoch][1], color='forestgreen', label="element2")
    else:
        plt.scatter(epoch, lambda_mu_cl3[epoch][0], color='lime')
        plt.scatter(epoch, lambda_mu_cl3[epoch][1], color='forestgreen')
    plt.xlim([0, Step])
    plt.ylabel("param class3")
    plt.xlabel("epoch")
    plt.legend()
    
    plt.pause(0.2)







