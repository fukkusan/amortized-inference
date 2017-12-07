import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_samples = 2
K = 3


# read csv
dataset_pi = pd.read_csv("res_pi.csv")

pi_cl1_r = []
pi_cl1_r.append( dataset_pi['pi_class1'] )
#print(pi_cl1_r)

pi_cl2_r = []
pi_cl2_r.append( dataset_pi['pi_class2'] )
#print(pi_cl2_r)

pi_cl3_r = []
pi_cl3_r.append( dataset_pi['pi_class3'] )
#print(pi_cl3_r)


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

pi_cl3 = []
start_pi_cl3 = 0
len_pi_cl3_r = len(pi_cl3_r[0])
while(start_pi_cl3 < len_pi_cl3_r):
    pi_cl3.append( np.mean( pi_cl3_r[0][start_pi_cl3:start_pi_cl3+num_samples-1] ) )
    start_pi_cl3 = start_pi_cl3 + num_samples
#print(pi_cl3)


# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(2):
    ax.cla()
    plt.bar([1, 2, 3], [ pi_cl1[epoch], pi_cl2[epoch], pi_cl3[epoch] ], width=0.5 )
    plt.xticks([1, 2, 3], ['class1', 'class2', 'class3'])
    
    plt.title("Mixing ratio")
    plt.ylabel("Mixing ratio")
    plt.ylim([0, 1])
    
    plt.pause(2.0)




