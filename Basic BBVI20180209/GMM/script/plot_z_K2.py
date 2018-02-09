import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_samples = 200
num_epochs = 1000
N = 1000
K = 2


# read csv
#dataset_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_z/200stepOK3/res_z_K2.csv")
dataset_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_pi_and_lambda_z/1000stepOK45/res_z_K2.csv")

z_cl1_r = []
z_cl1_r.append( dataset_z['z_class1'] )
#print(z_cl1_r)

z_cl2_r = []
z_cl2_r.append( dataset_z['z_class2'] )
#print(z_cl2_r)


# mean of samples
z_cl1 = []
for n in range(N):
    z_cl1.append(z_cl1_r[0][n])
z_cl1 = np.array(z_cl1)

z_cl2 = []
for n in range(N):
    z_cl2.append(z_cl2_r[0][n])
z_cl2 = np.array(z_cl2)

## mean of samples
#z_cl1 = []
#start_z_cl1 = 0
#len_z_cl1_r = len(z_cl1_r[0])
#while(start_z_cl1 < len_z_cl1_r):
#    z_cl1.append( np.mean( z_cl1_r[0][start_z_cl1:start_z_cl1+num_samples-1] ) )
#    start_z_cl1 = start_z_cl1 + num_samples
#z_cl1 = 1.0*(z_cl1>=np.ones(shape=N)*0.5)
#print(z_cl1)
#
#z_cl2 = []
#start_z_cl2 = 0
#len_z_cl2_r = len(z_cl2_r[0])
#while(start_z_cl2 < len_z_cl2_r):
#    z_cl2.append( np.mean( z_cl2_r[0][start_z_cl2:start_z_cl2+num_samples-1] ) )
#    start_z_cl2 = start_z_cl2 + num_samples
#z_cl2 = 1.0*(z_cl2 > np.ones(shape=N)*0.5)
#print(z_cl2)


## pile up N
#z_cl1_pile = []
#start1 = 0
#len_z_cl1 = len(z_cl1)
#while(start1 < len_z_cl1):
#    z_cl1_pile.append( np.sum( z_cl1[start1:start1+N-1] ) )
#    start1 = start1 + N
#print(z_cl1_pile)
#
#z_cl2_pile = []
#start2 = 0
#len_z_cl2 = len(z_cl2)
#while(start2 < len_z_cl2):
#    z_cl2_pile.append( np.sum( z_cl2[start2:start2+N-1] ) )
#    start2 = start2 + N
#print(z_cl2_pile)

data_frame = pd.DataFrame(index=[], columns=["z_cl1", "z_cl2"])
for n in range(N):
    series = pd.Series([z_cl1[n], z_cl2[n]], index=data_frame.columns)
    data_frame = data_frame.append(series, ignore_index=True)
data_frame.to_csv("res_plot_z_K2.csv", index= False)

# hist
fig = plt.figure()
ax = plt.axes()
#for epoch in range(num_epochs):
#ax.cla()
plt.bar([1, 2], [np.sum(z_cl1), np.sum(z_cl2)], width = 0.5, align='center')
plt.xticks([1, 2], ['class1', 'class2'])
plt.ylim([0, N])

plt.title("latent variables")
plt.ylabel("sum of N's latent variables")

#plt.pause(0.2)
plt.show()

