import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Gaussian_plot as gp
import os



# constant
num_samples = 10
num_epochs = 1000


# read csv
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver1.0/res_mu_and_Sigma_single_Gauss.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver1.0/gauss_data.csv")
dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver2.1/1000stepOK4/res_mu_and_Sigma_single_Gauss.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver2.1/1000stepOK4/gauss_data_ver2.1.csv")



mu_el1 = []
mu_el1.append( dataset_mu_and_Sigma['mu_element1'] )
print(mu_el1)                
mu_el2 = []                   
mu_el2.append( dataset_mu_and_Sigma['mu_element2'] )
print(mu_el2)                


Sigma_el11 = []
Sigma_el11.append(dataset_mu_and_Sigma['Sigma_element11'])
print(Sigma_el11)
Sigma_el12 = []
Sigma_el12.append(dataset_mu_and_Sigma['Sigma_element12'])
print(Sigma_el12)
Sigma_el21 = []
Sigma_el21.append(dataset_mu_and_Sigma['Sigma_element21'])
print(Sigma_el21)
Sigma_el22 = []
Sigma_el22.append(dataset_mu_and_Sigma['Sigma_element22'])
print(Sigma_el22)


observation_el1 = []
observation_el1.append( dataset_observation['element1'] )
observation_el2 = []
observation_el2.append( dataset_observation['element2'] )


# mean each classes
mu = []
start_mu = 0
len_mu = len(mu_el1[0])
while(start_mu < len_mu):
    mu.append( [ np.mean( mu_el1[0][start_mu:start_mu+num_samples-1] ), np.mean( mu_el2[0][start_mu:start_mu+num_samples-1] ) ] )
    start_mu = start_mu + num_samples
mu = np.array(mu)
print(mu)


# variance each classes
Sigma = []
start_Sigma = 0
len_Sigma = len(Sigma_el11[0])
while(start_Sigma < len_Sigma):
    Sigma.append([[np.mean(Sigma_el11[0][start_Sigma:start_Sigma+num_samples-1]), np.mean(Sigma_el12[0][start_Sigma:start_Sigma+num_samples-1])], [np.mean(Sigma_el21[0][start_Sigma:start_Sigma+num_samples-1]), np.mean(Sigma_el22[0][start_Sigma:start_Sigma+num_samples-1])]])
    start_Sigma = start_Sigma + num_samples
Sigma = np.array(Sigma)
print(Sigma)


# observation
observation = []
for n in range(len(observation_el1[0])):
    observation.append( [ observation_el1[0][n], observation_el2[0][n] ] )
observation = np.array(observation)
#print(observation)



# To csv
data_frame_mu = pd.DataFrame(index=[], columns=['mu_el1', 'mu_el2'])
for epoch in range(num_epochs):
    series_mu = pd.Series([mu[epoch][0], mu[epoch][1]], index=data_frame_mu.columns)
    data_frame_mu = data_frame_mu.append(series_mu, ignore_index=True)
data_frame_mu.to_csv("sample_mu.csv")



# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(num_epochs):
    ax.cla()
    #start = 0
    gp.plot_observations(ax, observation, "k")
    gp.plot_mean_variance(ax, mu[epoch], Sigma[epoch], 'b', scale=100.0)
        
    #plt.xlim([-100, 100])
    #plt.ylim([-100, 100])
    plt.title("Single Gaussian on 2-dim space")
    plt.xlabel("x1-axis")
    plt.ylabel("x2-axis")
    
    plt.pause(0.2)





