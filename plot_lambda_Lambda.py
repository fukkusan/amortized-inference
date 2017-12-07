import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
K = 3
Step = 2


# read csv
dataset_lambda_Lambda = pd.read_csv("res_lambda_Lambda.csv")

lambda_Lambda_cl1_el11_r = []
lambda_Lambda_cl1_el11_r.append( dataset_lambda_Lambda['lambda_Lambda_class1_element11'] )
#print(lambda_Lambda_cl1_el11_r)

lambda_Lambda_cl1_el12_r = []
lambda_Lambda_cl1_el12_r.append( dataset_lambda_Lambda['lambda_Lambda_class1_element12'] )
#print(lambda_Lambda_cl1_el12_r)

lambda_Lambda_cl1_el21_r = []
lambda_Lambda_cl1_el21_r.append( dataset_lambda_Lambda['lambda_Lambda_class1_element21'] )
#print(lambda_Lambda_cl1_el21_r)

lambda_Lambda_cl1_el22_r = []
lambda_Lambda_cl1_el22_r.append( dataset_lambda_Lambda['lambda_Lambda_class1_element22'] )
#print(lambda_Lambda_cl1_el22_r)

lambda_Lambda_cl2_el11_r = []
lambda_Lambda_cl2_el11_r.append( dataset_lambda_Lambda['lambda_Lambda_class2_element11'] )
#print(lambda_Lambda_cl2_el11_r)

lambda_Lambda_cl2_el12_r = []
lambda_Lambda_cl2_el12_r.append( dataset_lambda_Lambda['lambda_Lambda_class2_element12'] )
#print(lambda_Lambda_cl2_el12_r)

lambda_Lambda_cl2_el21_r = []
lambda_Lambda_cl2_el21_r.append( dataset_lambda_Lambda['lambda_Lambda_class2_element21'] )
#print(lambda_Lambda_cl2_el21_r)

lambda_Lambda_cl2_el22_r = []
lambda_Lambda_cl2_el22_r.append( dataset_lambda_Lambda['lambda_Lambda_class2_element22'] )
#print(lambda_Lambda_cl2_el22_r)

lambda_Lambda_cl3_el11_r = []
lambda_Lambda_cl3_el11_r.append( dataset_lambda_Lambda['lambda_Lambda_class3_element11'] )
#print(lambda_Lambda_cl3_el11_r)

lambda_Lambda_cl3_el12_r = []
lambda_Lambda_cl3_el12_r.append( dataset_lambda_Lambda['lambda_Lambda_class3_element12'] )
#print(lambda_Lambda_cl3_el12_r)

lambda_Lambda_cl3_el21_r = []
lambda_Lambda_cl3_el21_r.append( dataset_lambda_Lambda['lambda_Lambda_class3_element21'] )
#print(lambda_Lambda_cl3_el21_r)

lambda_Lambda_cl3_el22_r = []
lambda_Lambda_cl3_el22_r.append( dataset_lambda_Lambda['lambda_Lambda_class3_element22'] )
#print(lambda_Lambda_cl3_el22_r)


lambda_Lambda_cl1 = []
for step in range(Step):
    lambda_Lambda_cl1.append( [ [ lambda_Lambda_cl1_el11_r[0][step], lambda_Lambda_cl1_el12_r[0][step] ], [ lambda_Lambda_cl1_el21_r[0][step], lambda_Lambda_cl1_el22_r[0][step] ] ] )
#print(lambda_Lambda_cl1)

lambda_Lambda_cl2 = []
for step in range(Step):
    lambda_Lambda_cl2.append( [ [ lambda_Lambda_cl2_el11_r[0][step], lambda_Lambda_cl2_el12_r[0][step] ], [ lambda_Lambda_cl2_el21_r[0][step], lambda_Lambda_cl2_el22_r[0][step] ] ] )
#print(lambda_Lambda_cl2)

lambda_Lambda_cl3 = []
for step in range(Step):
    lambda_Lambda_cl3.append( [ [ lambda_Lambda_cl3_el11_r[0][step], lambda_Lambda_cl3_el12_r[0][step] ], [ lambda_Lambda_cl3_el21_r[0][step], lambda_Lambda_cl3_el22_r[0][step] ] ] )
#print(lambda_Lambda_cl3)


# plot
eig_vals_lambda_Lambda_cl1 = np.linalg.eig(lambda_Lambda_cl1)[0]
#print(eig_vals_lambda_Lambda_cl1)
eig_vals_lambda_Lambda_cl2 = np.linalg.eig(lambda_Lambda_cl2)[0]
#print(eig_vals_lambda_Lambda_cl2)
eig_vals_lambda_Lambda_cl3 = np.linalg.eig(lambda_Lambda_cl3)[0]
#print(eig_vals_lambda_Lambda_cl3)
fig = plt.figure()
ax = plt.axes()
for epoch in range(Step):
    ax.cla()
    plt.subplot(K, 1, 1)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl1[epoch][0], color='cyan', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl1[epoch][1], color='midnightblue', label="2nd eigen value")
    else:
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl1[epoch][0], color='cyan')
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl1[epoch][1], color='midnightblue')
    plt.ylabel("eig_vals class1")
    plt.xlim([0, Step])
    plt.legend()
    plt.subplot(K, 1, 2)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl2[epoch][0], color='salmon', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl2[epoch][1], color='tomato', label="2nd eigen value")
    else:
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl2[epoch][0], color='salmon')
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl2[epoch][1], color='tomato')
    plt.ylabel("eig_vals class2")
    plt.xlim([0, Step])
    plt.legend()
    plt.subplot(K, 1, 3)
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl3[epoch][0], color='lime', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl3[epoch][1], color='forestgreen', label="2nd eigne value")
    else:
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl3[epoch][0], color='lime')
        plt.scatter(epoch, eig_vals_lambda_Lambda_cl3[epoch][1], color='forestgreen')
    plt.xlabel("epoch")
    plt.ylabel("eig_vals class3")
    plt.xlim([0, Step])
    plt.legend()
    plt.suptitle("Eigen values of lambda_Lambda")
    
    
    plt.pause(1.2)
plt.close()



