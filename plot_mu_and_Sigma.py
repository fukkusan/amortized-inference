import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Gaussian_plot as gp
import os



# constant
num_samples = 10
num_epochs = 100


# read csv
dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_mu_and_Sigma.csv")
dataset_observation = pd.read_csv("gmm_data.csv")

mu_cl1_el1 = []
mu_cl1_el1.append( dataset_mu_and_Sigma['mu_class1_element1'] )
#print(mu_cl1_el1)                
mu_cl1_el2 = []                   
mu_cl1_el2.append( dataset_mu_and_Sigma['mu_class1_element2'] )
#print(mu_cl1_el2)                
mu_cl2_el1 = []                   
mu_cl2_el1.append( dataset_mu_and_Sigma['mu_class2_element1'] )
#print(mu_cl2_el1)                
mu_cl2_el2 = []                   
mu_cl2_el2.append( dataset_mu_and_Sigma['mu_class2_element2'] )
#print(mu_cl2_el2)                
mu_cl3_el1 = []                   
mu_cl3_el1.append( dataset_mu_and_Sigma['mu_class3_element1'] )
#print(mu_cl3_el1)                
mu_cl3_el2 = []                   
mu_cl3_el2.append( dataset_mu_and_Sigma['mu_class3_element2'] )
#print(mu_cl3_el2)

Sigma_cl1_el11 = []
Sigma_cl1_el11.append( dataset_mu_and_Sigma['Sigma_class1_element11'] )
#print(Sigma_cl1_el11)
Sigma_cl1_el12 = []
Sigma_cl1_el12.append( dataset_mu_and_Sigma['Sigma_class1_element12'] )
#print(Sigma_cl1_el12)
Sigma_cl1_el21 = []
Sigma_cl1_el21.append( dataset_mu_and_Sigma['Sigma_class1_element21'] )
#print(Sigma_cl1_el21)
Sigma_cl1_el22 = []
Sigma_cl1_el22.append( dataset_mu_and_Sigma['Sigma_class1_element22'] )
#print(Sigma_cl1_el22)
Sigma_cl2_el11 = []
Sigma_cl2_el11.append( dataset_mu_and_Sigma['Sigma_class2_element11'] )
#print(Sigma_cl2_el11)
Sigma_cl2_el12 = []
Sigma_cl2_el12.append( dataset_mu_and_Sigma['Sigma_class2_element12'] )
#print(Sigma_cl2_el12)
Sigma_cl2_el21 = []
Sigma_cl2_el21.append( dataset_mu_and_Sigma['Sigma_class2_element21'] )
#print(Sigma_cl2_el21)
Sigma_cl2_el22 = []
Sigma_cl2_el22.append( dataset_mu_and_Sigma['Sigma_class2_element22'] )
#print(Sigma_cl2_el22)
Sigma_cl3_el11 = []
Sigma_cl3_el11.append( dataset_mu_and_Sigma['Sigma_class3_element11'] )
#print(Sigma_cl3_el11)
Sigma_cl3_el12 = []
Sigma_cl3_el12.append( dataset_mu_and_Sigma['Sigma_class3_element12'] )
#print(Sigma_cl3_el12)
Sigma_cl3_el21 = []
Sigma_cl3_el21.append( dataset_mu_and_Sigma['Sigma_class3_element21'] )
#print(Sigma_cl3_el21)
Sigma_cl3_el22 = []
Sigma_cl3_el22.append( dataset_mu_and_Sigma['Sigma_class3_element22'] )
#print(Sigma_cl3_el22)

observation_el1 = []
observation_el1.append( dataset_observation['element1'] )
observation_el2 = []
observation_el2.append( dataset_observation['element2'] )


# mean each classes
#mu_cl1_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        mu_cl1_sam.append( [ mu_cl1_el1[0][num_samples*i + s], mu_cl1_el2[0][num_samples*i + s] ] )
#print(mu_cl1_sam)
mu_cl1 = []
start_mc1 = 0
len_mu_cl1 = len(mu_cl1_el1[0])
while(start_mc1 < len_mu_cl1):
    mu_cl1.append( [ np.mean( mu_cl1_el1[0][start_mc1:start_mc1+num_samples-1] ), np.mean( mu_cl1_el2[0][start_mc1:start_mc1+num_samples-1] ) ] )
    start_mc1 = start_mc1 + num_samples
mu_cl1 = np.array(mu_cl1)
#print(mu_cl1)

#mu_cl2_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        mu_cl2_sam.append( [ mu_cl2_el1[0][num_samples*i + s], mu_cl2_el2[0][num_samples*i + s] ] )
#mu_cl2_sam = np.array(mu_cl2_sam)
#print(mu_cl2_sam)
mu_cl2 = []
start_mc2 = 0
len_mu_cl2 = len(mu_cl2_el1[0])
while(start_mc2 < len_mu_cl2):
    mu_cl2.append( [ np.mean( mu_cl2_el1[0][start_mc2:start_mc1+num_samples-1] ), np.mean( mu_cl2_el2[0][start_mc2:start_mc2+num_samples-1] ) ] )
    start_mc2 = start_mc2 + num_samples
mu_cl2 = np.array(mu_cl2)
#print(mu_cl2)

#mu_cl3_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        mu_cl3_sam.append( [ mu_cl3_el1[0][num_samples*i + s], mu_cl3_el2[0][num_samples*i + s] ] )
#mu_cl3_sam = np.array(mu_cl3_sam)
#print(mu_cl3_sam)
mu_cl3 = []
start_mc3 = 0
len_mu_cl3 = len(mu_cl3_el1[0])
while(start_mc3 < len_mu_cl3):
    mu_cl3.append( [ np.mean( mu_cl3_el1[0][start_mc3:start_mc3+num_samples-1] ), np.mean( mu_cl3_el2[0][start_mc3:start_mc3+num_samples-1] ) ] )
    start_mc3 = start_mc3 + num_samples
mu_cl3 = np.array(mu_cl3)
#print(mu_cl3)


# variance each classes
#Sigma_cl1_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        Sigma_cl1_sam.append( [ [ Sigma_cl1_el11[0][num_samples*i + s], Sigma_cl1_el12[0][num_samples*i + s] ], [ Sigma_cl1_el21[0][num_samples*i + s], Sigma_cl1_el22[0][num_samples*i + s] ] ] )
#Sigma_cl1_sam = np.array(Sigma_cl1_sam)
#print(Sigma_cl1_sam)
Sigma_cl1 = []
start_Sigma1 = 0
len_Sigma1 = len(Sigma_cl1_el11[0])
while(start_Sigma1 < len_Sigma1):
    Sigma_cl1.append( [ [ np.mean( Sigma_cl1_el11[0][start_Sigma1:start_Sigma1+num_samples-1] ), np.mean( Sigma_cl1_el12[0][start_Sigma1:start_Sigma1+num_samples-1] ) ], [ np.mean( Sigma_cl1_el21[0][start_Sigma1:start_Sigma1+num_samples-1] ), np.mean( Sigma_cl1_el22[0][start_Sigma1:start_Sigma1+num_samples-1] ) ] ] )
    start_Sigma1 = start_Sigma1 + num_samples
Sigma_cl1 = np.array(Sigma_cl1)
print(Sigma_cl1)

#Sigma_cl2_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        Sigma_cl2_sam.append( [ [ Sigma_cl2_el11[0][num_samples*i + s], Sigma_cl2_el12[0][num_samples*i + s] ], [ Sigma_cl2_el21[0][num_samples*i + s], Sigma_cl2_el22[0][num_samples*i + s] ] ] )
#Sigma_cl2_sam = np.array(Sigma_cl2_sam)
#print(Sigma_cl2_sam)
Sigma_cl2 = []
start_Sigma2 = 0
len_Sigma2 = len(Sigma_cl2_el11[0])
while(start_Sigma2 < len_Sigma2):
    Sigma_cl2.append( [ [ np.mean( Sigma_cl2_el11[0][start_Sigma2:start_Sigma2+num_samples-1] ), np.mean( Sigma_cl2_el12[0][start_Sigma2:start_Sigma2+num_samples-1] ) ], [ np.mean( Sigma_cl2_el21[0][start_Sigma2:start_Sigma2+num_samples-1] ), np.mean( Sigma_cl2_el22[0][start_Sigma2:start_Sigma2+num_samples-1] ) ] ] )
    start_Sigma2 = start_Sigma2 + num_samples
Sigma_cl2 = np.array(Sigma_cl2)
print(Sigma_cl2)

#Sigma_cl3_sam = []
#for s in range(num_samples):
#    for i in range(num_samples):
#        Sigma_cl3_sam.append( [ [ Sigma_cl3_el11[0][num_samples*i + s], Sigma_cl3_el12[0][num_samples*i + s] ], [ Sigma_cl3_el21[0][num_samples*i + s], Sigma_cl3_el22[0][num_samples*i + s] ] ] )
#Sigma_cl3_sam = np.array(Sigma_cl3_sam)
#print(Sigma_cl3_sam)
Sigma_cl3 = []
start_Sigma3 = 0
len_Sigma3 = len(Sigma_cl3_el11[0])
while(start_Sigma3 < len_Sigma3):
    Sigma_cl3.append( [ [ np.mean( Sigma_cl3_el11[0][start_Sigma3:start_Sigma3+num_samples-1] ), np.mean( Sigma_cl3_el12[0][start_Sigma3:start_Sigma3+num_samples-1] ) ], [ np.mean( Sigma_cl3_el21[0][start_Sigma3:start_Sigma3+num_samples-1] ), np.mean( Sigma_cl3_el22[0][start_Sigma3:start_Sigma3+num_samples-1] ) ] ] )
    start_Sigma3 = start_Sigma3 + num_samples
Sigma_cl3 = np.array(Sigma_cl3)
print(Sigma_cl3)

# observation
observation = []
for n in range(len(observation_el1[0])):
    observation.append( [ observation_el1[0][n], observation_el2[0][n] ] )
observation = np.array(observation)
#print(observation)



# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(num_epochs):
    ax.cla()
    #start = 0
    gp.plot_observations(ax, observation, "k")
    gp.plot_mean_variance(ax, mu_cl1[epoch], Sigma_cl1[epoch], 'b', scale=0.1)
    gp.plot_mean_variance(ax, mu_cl2[epoch], Sigma_cl2[epoch], 'r', scale=0.1)
    gp.plot_mean_variance(ax, mu_cl3[epoch], Sigma_cl3[epoch], 'g', scale=0.1)
#    #while (start<=num_samples):
#    #   gp.plot_mean_variance(ax, mu_cl1_sam[start:start+num_samples][epoch], Sigma_cl1_sam[:num_samples][epoch], 'b', scale=0.1)
#    #   gp.plot_mean_variance(ax, mu_cl2_sam[start:start+num_samples][epoch], Sigma_cl2_sam[:num_samples][epoch], 'r', scale=0.1)
#    #   gp.plot_mean_variance(ax, mu_cl3_sam[start:start+num_samples][epoch], Sigma_cl3_sam[:num_samples][epoch], 'g', scale=0.1)
#    #   start = start + num_samples
#       
    #plt.xlim([-100, 100])
    #plt.ylim([-100, 100])
    plt.title("Gaussian on 2-dim space")
    plt.xlabel("x1-axis")
    plt.ylabel("x2-axis")
    
    plt.pause(0.2)





