import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_samples = 100
num_epochs = 1000
K = 2


# read csv
#dataset_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_pi.csv")
#dataset_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/res_pi_K2.csv")
dataset_pi = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S100epoch1000sample100/1000stepOK10/res_pi_K2.csv")

pi_cl1_r = []
pi_cl1_r.append( dataset_pi['pi_class1'] )
#print(pi_cl1_r)

pi_cl2_r = []
pi_cl2_r.append( dataset_pi['pi_class2'] )
#print(pi_cl2_r)


pi_cl1 = []
start_pi_cl1 = 0
len_pi_cl1_r = len(pi_cl1_r[0])
while(start_pi_cl1 < len_pi_cl1_r):
    pi_cl1.append( np.mean( pi_cl1_r[0][start_pi_cl1:start_pi_cl1+num_samples-1] ) )
    start_pi_cl1 = start_pi_cl1 + num_samples
#print(pi_cl1)

pi_cl2 = []
start_pi_cl2 = 0
len_pi_cl2_r = len(pi_cl2_r[0])
while(start_pi_cl2 < len_pi_cl2_r):
    pi_cl2.append( np.mean( pi_cl2_r[0][start_pi_cl2:start_pi_cl2+num_samples-1] ) )
    start_pi_cl2 = start_pi_cl2 + num_samples
#print(pi_cl2)



# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(num_epochs):
    ax.cla()
    plt.bar([1, 2], [pi_cl1[epoch], pi_cl2[epoch]], width=0.5 )
    plt.xticks([1, 2], ['class1', 'class2'])
    
    plt.title("Mixing ratio")
    plt.ylabel("Mixing ratio")
    plt.ylim([0, 1])
    
    #plt.pause(0.2)
plt.show()



