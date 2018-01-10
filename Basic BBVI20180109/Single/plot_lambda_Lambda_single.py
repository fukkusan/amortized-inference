import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
Step = 1000


# read csv
dataset_lambda_Lambda = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver2.1/1000stepOK4/res_lambda_Lambda_single_Gauss.csv")

lambda_Lambda_el11_r = []
lambda_Lambda_el11_r.append( dataset_lambda_Lambda['lambda_Lambda_element11'] )
#print(lambda_Lambda_cl1_el11_r)

lambda_Lambda_el12_r = []
lambda_Lambda_el12_r.append( dataset_lambda_Lambda['lambda_Lambda_element12'] )
#print(lambda_Lambda_cl1_el12_r)

lambda_Lambda_el21_r = []
lambda_Lambda_el21_r.append( dataset_lambda_Lambda['lambda_Lambda_element21'] )
#print(lambda_Lambda_cl1_el21_r)

lambda_Lambda_el22_r = []
lambda_Lambda_el22_r.append( dataset_lambda_Lambda['lambda_Lambda_element22'] )
#print(lambda_Lambda_cl1_el22_r)



lambda_Lambda = []
for step in range(Step):
    lambda_Lambda.append( [ [ lambda_Lambda_el11_r[0][step], lambda_Lambda_el12_r[0][step] ], [ lambda_Lambda_el21_r[0][step], lambda_Lambda_el22_r[0][step] ] ] )
#print(lambda_Lambda_cl1)



# plot
eig_vals_lambda_Lambda = np.linalg.eig(lambda_Lambda)[0]
#print(eig_vals_lambda_Lambda_cl1)
fig = plt.figure()
ax = plt.axes()
for epoch in range(Step):
    #ax.cla()
    if(epoch==0):
        plt.scatter(epoch, eig_vals_lambda_Lambda[epoch][0], color='cyan', label="1st eigen value")
        plt.scatter(epoch, eig_vals_lambda_Lambda[epoch][1], color='midnightblue', label="2nd eigen value")
    else:
        plt.scatter(epoch, eig_vals_lambda_Lambda[epoch][0], color='cyan')
        plt.scatter(epoch, eig_vals_lambda_Lambda[epoch][1], color='midnightblue')
    plt.ylabel("eig_vals")
    plt.xlim([0, Step])
    plt.legend()
    plt.title("Eigen values of lambda_Lambda")
    
    
plt.show()
plt.close()



