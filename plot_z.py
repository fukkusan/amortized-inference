import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_samples = 2
N = 10
K = 3


# read csv
dataset_z = pd.read_csv("res_z.csv")

z_cl1_r = []
z_cl1_r.append( dataset_z['z_class1'] )
#print(z_cl1_r)

z_cl2_r = []
z_cl2_r.append( dataset_z['z_class2'] )
#print(z_cl2_r)

z_cl3_r = []
z_cl3_r.append( dataset_z['z_class3'] )
#print(z_cl3_r)


# mean of samples
z_cl1 = []
start_z_cl1 = 0
len_z_cl1_r = len(z_cl1_r[0])
while(start_z_cl1 < len_z_cl1_r):
    z_cl1.append( np.mean( z_cl1_r[0][start_z_cl1:start_z_cl1+num_samples-1] ) )
    start_z_cl1 = start_z_cl1 + num_samples
#print(z_cl1)

z_cl2 = []
start_z_cl2 = 0
len_z_cl2_r = len(z_cl2_r[0])
while(start_z_cl2 < len_z_cl2_r):
    z_cl2.append( np.mean( z_cl2_r[0][start_z_cl2:start_z_cl2+num_samples-1] ) )
    start_z_cl2 = start_z_cl2 + num_samples
#print(z_cl2)

z_cl3 = []
start_z_cl3 = 0
len_z_cl3_r = len(z_cl3_r[0])
while(start_z_cl3 < len_z_cl3_r):
    z_cl3.append( np.mean( z_cl3_r[0][start_z_cl3:start_z_cl3+num_samples-1] ) )
    start_z_cl3 = start_z_cl3 + num_samples
#print(z_cl3)


# pile up N
z_cl1_pile = []
start1 = 0
len_z_cl1 = len(z_cl1)
while(start1 < len_z_cl1):
    z_cl1_pile.append( np.sum( z_cl1[start1:start1+N-1] ) )
    start1 = start1 + N
print(z_cl1_pile)

z_cl2_pile = []
start2 = 0
len_z_cl2 = len(z_cl2)
while(start2 < len_z_cl2):
    z_cl2_pile.append( np.sum( z_cl2[start2:start2+N-1] ) )
    start2 = start2 + N
print(z_cl2_pile)

z_cl3_pile = []
start3 = 0
len_z_cl3 = len(z_cl3)
while(start3 < len_z_cl3):
    z_cl3_pile.append( np.sum( z_cl3[start3:start3+N-1] ) )
    start3 = start3 + N
print(z_cl3_pile)


# hist
fig = plt.figure()
ax = plt.axes()
for epoch in range(2):
    ax.cla()
    plt.bar( [1, 2, 3], [ z_cl1_pile[epoch], z_cl2_pile[epoch], z_cl3_pile[epoch] ], width = 0.5, align='center' )
    plt.xticks([1, 2, 3], ['class1', 'class2', 'class3'])
    plt.ylim([0, N])
    
    plt.title("latent variables")
    plt.ylabel("sum of N's latent variables")
    
    plt.pause(1.0)

