import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Gaussian_plot as gp



# constant
K = 3
Step = 100



# read csv
dataset_lambda_muLambda = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_lambda_muLambda.csv")


lambda_muLambda_cl1_el11 = []
lambda_muLambda_cl1_el11.append( dataset_lambda_muLambda['lambda_muLambda_class1_element11'] )
#print(lambda_muLambda_cl1_el11)

lambda_muLambda_cl1_el12 = []
lambda_muLambda_cl1_el12.append( dataset_lambda_muLambda['lambda_muLambda_class1_element12'] )
#print(lambda_muLambda_cl1_el12)

lambda_muLambda_cl1_el21 = []
lambda_muLambda_cl1_el21.append( dataset_lambda_muLambda['lambda_muLambda_class1_element21'] )
#print(lambda_muLambda_cl1_el21)

lambda_muLambda_cl1_el22 = []
lambda_muLambda_cl1_el22.append( dataset_lambda_muLambda['lambda_muLambda_class1_element22'] )
#print(lambda_muLambda_cl1_el22)

lambda_muLambda_cl2_el11 = []
lambda_muLambda_cl2_el11.append( dataset_lambda_muLambda['lambda_muLambda_class2_element11'] )
#print(lambda_muLambda_cl2_el11)

lambda_muLambda_cl2_el12 = []
lambda_muLambda_cl2_el12.append( dataset_lambda_muLambda['lambda_muLambda_class2_element12'] )
#print(lambda_muLambda_cl2_el12)

lambda_muLambda_cl2_el21 = []
lambda_muLambda_cl2_el21.append( dataset_lambda_muLambda['lambda_muLambda_class2_element21'] )
#print(lambda_muLambda_cl2_el21)

lambda_muLambda_cl2_el22 = []
lambda_muLambda_cl2_el22.append( dataset_lambda_muLambda['lambda_muLambda_class2_element22'] )
#print(lambda_muLambda_cl2_el22)

lambda_muLambda_cl3_el11 = []
lambda_muLambda_cl3_el11.append( dataset_lambda_muLambda['lambda_muLambda_class3_element11'] )
#print(lambda_muLambda_cl3_el11)

lambda_muLambda_cl3_el12 = []
lambda_muLambda_cl3_el12.append( dataset_lambda_muLambda['lambda_muLambda_class3_element12'] )
#print(lambda_muLambda_cl3_el12)

lambda_muLambda_cl3_el21 = []
lambda_muLambda_cl3_el21.append( dataset_lambda_muLambda['lambda_muLambda_class3_element21'] )
#print(lambda_muLambda_cl3_el21)

lambda_muLambda_cl3_el22 = []
lambda_muLambda_cl3_el22.append( dataset_lambda_muLambda['lambda_muLambda_class3_element22'] )
#print(lambda_muLambda_cl3_el22)


# variance each classes
lambda_muLambda_cl1 = []
for step in range(Step):
    lambda_muLambda_cl1.append( [ [ lambda_muLambda_cl1_el11[0][step], lambda_muLambda_cl1_el12[0][step] ], [ lambda_muLambda_cl1_el21[0][step], lambda_muLambda_cl1_el22[0][step] ] ] )
#print(lambda_muLambda_cl1)

lambda_muLambda_cl2 = []
for step in range(Step):
    lambda_muLambda_cl2.append( [ [ lambda_muLambda_cl2_el11[0][step], lambda_muLambda_cl2_el12[0][step] ], [ lambda_muLambda_cl2_el21[0][step], lambda_muLambda_cl2_el22[0][step] ] ] )
#print(lambda_muLambda_cl2)

lambda_muLambda_cl3 = []
for step in range(Step):
    lambda_muLambda_cl3.append( [ [ lambda_muLambda_cl3_el11[0][step], lambda_muLambda_cl3_el12[0][step] ], [ lambda_muLambda_cl3_el21[0][step], lambda_muLambda_cl3_el22[0][step] ] ] )
#print(lambda_muLambda_cl3)


# plot
eig_vals_lambda_muLambda_cl1 = np.linalg.eig(lambda_muLambda_cl1)[0]
#print(eig_vals_lambda_muLambda_cl1)
eig_vals_lambda_muLambda_cl2 = np.linalg.eig(lambda_muLambda_cl2)[0]
#print(eig_vals_lambda_muLambda_cl2)
eig_vals_lambda_muLambda_cl3 = np.linalg.eig(lambda_muLambda_cl3)[0]
#print(eig_vals_lambda_muLambda_cl3)
fig = plt.figure()
ax = plt.axes()
for epoch in range(Step):
    ax.cla()
    plt.subplot(K, 1, 1)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl1[epoch][0], color='cyan', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl1[epoch][1], color='midnightblue', label="2nd eigen value")
    else:
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl1[epoch][0], color='cyan')
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl1[epoch][1], color='midnightblue')
    plt.ylabel("eig_vals class1")
    plt.xlim([0, Step])
    plt.legend()
    plt.subplot(K, 1, 2)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl2[epoch][0], color='salmon', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl2[epoch][1], color='tomato', label="2nd eigen value")
    else:
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl2[epoch][0], color='salmon')
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl2[epoch][1], color='tomato')
    plt.ylabel("eig_vals class2")
    plt.xlim([0, Step])
    plt.legend()
    plt.subplot(K, 1, 3)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl3[epoch][0], color='lime', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl3[epoch][1], color='forestgreen', label="2nd eigne value")
    else:
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl3[epoch][0], color='lime')
        plt.scatter(epoch, eig_vals_lambda_muLambda_cl3[epoch][1], color='forestgreen')
    plt.xlabel("epoch")
    plt.ylabel("eig_vals class3")
    plt.xlim([0, Step])
    plt.legend()
    plt.suptitle("Eigen values of lambda_muLambda")
    
    
    plt.pause(0.2)
plt.close()




