import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Gaussian_plot as gp
import os



# constant
num_samples = 100
num_epochs = 1000


# read csv
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_mu_and_Sigma.csv")
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.5_N1000S100epoch1000sample100/1000stepOK/res_mu_and_Sigma.csv")
#dataset_observation = pd.read_csv("gmm_data_ver2.2.5.csv")
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_N1000S100epoch1000sample100/1000stepOK3/res_mu_and_Sigma_K2.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_N1000S100epoch1000sample100/1000stepOK3/gmm_data_ver2.2.6.csv")
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/res_mu_and_Sigma_K2.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/gmm_data_ver2.2.6_ho.csv")
dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S100epoch1000sample100/1000stepOK10/res_mu_and_Sigma_K2.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S100epoch1000sample100/1000stepOK10/gmm_data_ver2.2.7.csv")
#dataset_mu_and_Sigma = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/no_care_N200S100epoch50sample100/res_mu_and_Sigma_K2_nocare.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/no_care_N200S100epoch50sample100/gmm_data_nocare.csv")



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


Sigma_cl1_el11 = []
Sigma_cl1_el11.append(dataset_mu_and_Sigma['Sigma_class1_element11'])
#print(Sigma_cl1_el11)
Sigma_cl1_el12 = []
Sigma_cl1_el12.append(dataset_mu_and_Sigma['Sigma_class1_element12'])
#print(Sigma_cl1_el12)
Sigma_cl1_el21 = []
Sigma_cl1_el21.append(dataset_mu_and_Sigma['Sigma_class1_element21'])
#print(Sigma_cl1_el21)
Sigma_cl1_el22 = []
Sigma_cl1_el22.append(dataset_mu_and_Sigma['Sigma_class1_element22'])
#print(Sigma_cl1_el22)
Sigma_cl2_el11 = []
Sigma_cl2_el11.append(dataset_mu_and_Sigma['Sigma_class2_element11'])
#print(Sigma_cl2_el11)
Sigma_cl2_el12 = []
Sigma_cl2_el12.append(dataset_mu_and_Sigma['Sigma_class2_element12'])
#print(Sigma_cl2_el12)
Sigma_cl2_el21 = []
Sigma_cl2_el21.append(dataset_mu_and_Sigma['Sigma_class2_element21'])
#print(Sigma_cl2_el21)
Sigma_cl2_el22 = []
Sigma_cl2_el22.append(dataset_mu_and_Sigma['Sigma_class2_element22'])
#print(Sigma_cl2_el22)


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



# observation
observation = []
for n in range(len(observation_el1[0])):
    observation.append( [ observation_el1[0][n], observation_el2[0][n] ] )
observation = np.array(observation)
#print(observation)



# To csv
data_frame_mu = pd.DataFrame(index=[], columns=['mu_cl1_el1', 'mu_cl1_el2', 'mu_cl2_el1', 'mu_cl2_el2'])
for epoch in range(num_epochs):
    series_mu = pd.Series([mu_cl1[epoch][0], mu_cl1[epoch][1], mu_cl2[epoch][0], mu_cl2[epoch][1]], index=data_frame_mu.columns)
    data_frame_mu = data_frame_mu.append(series_mu, ignore_index=True)
data_frame_mu.to_csv("sample_mu_K2.csv")



# plot
fig = plt.figure()
ax = plt.axes()
for epoch in range(num_epochs):
    ax.cla()
    #start = 0
    gp.plot_observations(ax, observation, "k")
    gp.plot_mean_variance(ax, mu_cl1[epoch], Sigma_cl1[epoch], 'b', scale=10.0)		# ver2.2.6_ho 50.0
    gp.plot_mean_variance(ax, mu_cl2[epoch], Sigma_cl2[epoch], 'r', scale=10.0)		# ver2.2.6_ho 50.0
#    #while (start<=num_samples):
#    #   gp.plot_mean_variance(ax, mu_cl1_sam[start:start+num_samples][epoch], Sigma_cl1_sam[:num_samples][epoch], 'b', scale=100.0)
#    #   gp.plot_mean_variance(ax, mu_cl2_sam[start:start+num_samples][epoch], Sigma_cl2_sam[:num_samples][epoch], 'r', scale=100.0)
#    #   gp.plot_mean_variance(ax, mu_cl3_sam[start:start+num_samples][epoch], Sigma_cl3_sam[:num_samples][epoch], 'g', scale=100.0)
#    #   start = start + num_samples
#       
    #plt.xlim([-100, 100])
    #plt.ylim([-100, 100])
    plt.title("Gaussian on 2-dim space")
    plt.xlabel("x1-axis")
    plt.ylabel("x2-axis")
    
    plt.pause(0.1)





