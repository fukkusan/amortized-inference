import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import Gaussian_plot2 as gp
import custumize_colormap as cc



# constant
num_epochs = 200
N = 1000
K = 2
D = 2


# read csv
#dataset_lambda_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_pi_and_lambda_z/200stepOK37/res_lambda_z_K2.csv")
#dataset_generated_data = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_pi_and_lambda_z/200stepOK37/res_generated_data.csv")
#dataset_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_pi_and_lambda_z/200stepOK37/res_z_K2.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_pi_and_lambda_z/200stepOK37/gmm_data.csv")
dataset_lambda_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_z/200stepOK3/res_lambda_z_K2.csv")
dataset_generated_data = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_z/200stepOK3/res_generated_data.csv")
dataset_z = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_z/200stepOK3/res_z_K2.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver_lambda_z/200stepOK3/gmm_data.csv")

observation_el1_r = []
observation_el1_r.append( dataset_observation['element1'] )
observation_el2_r = []
observation_el2_r.append( dataset_observation['element2'] )

lambda_z_cl1_r = []
lambda_z_cl1_r.append(dataset_lambda_z['lambda_z_class1'])
#print(lambda_z_cl1_r)
lambda_z_cl2_r = []
lambda_z_cl2_r.append(dataset_lambda_z['lambda_z_class2'])
#print(lambda_z_cl2_r)

generated_data_cl1_el1_r = []
generated_data_cl1_el1_r.append(dataset_generated_data['generated_data_cl1_el1'])
#print(generated_data_cl1_el1_r)
generated_data_cl1_el2_r = []
generated_data_cl1_el2_r.append(dataset_generated_data['generated_data_cl1_el2'])
#print(generated_data_cl1_el2_r)
generated_data_cl2_el1_r = []
generated_data_cl2_el1_r.append(dataset_generated_data['generated_data_cl2_el1'])
#print(generated_data_cl2_el1_r)
generated_data_cl2_el2_r = []
generated_data_cl2_el2_r.append(dataset_generated_data['generated_data_cl2_el2'])
#print(generated_data_cl2_el2_r)

z_cl1_r = []
z_cl1_r.append(dataset_z['z_class1'])
#print(z_cl1_r)
z_cl2_r = []
z_cl2_r.append(dataset_z['z_class2'])
#print(z_cl2_r)


# observation
observation = []
for n in range(len(observation_el1_r[0])):
    observation.append([observation_el1_r[0][n], observation_el2_r[0][n]])
observation = np.array(observation)
#print(observation)

observation_el1 = []
for n in range(len(observation_el1_r[0])):
    observation_el1.append(observation_el1_r[0][n])
observation_el1 = np.array(observation_el1)
print(observation_el1)

observation_el2 = []
for n in range(len(observation_el2_r[0])):
    observation_el2.append(observation_el2_r[0][n])
observation_el2 = np.array(observation_el2)
print(observation_el2)



# lambda_z(assocoate degrees)
lambda_z_cl1 = []
lambda_z_cl2 = []
for n in range(N):
    lambda_z_cl1.append(lambda_z_cl1_r[0][n])
    lambda_z_cl2.append(lambda_z_cl2_r[0][n])
#lambda_z_cl1 = np.round(np.array(lambda_z_cl1), 3)
#lambda_z_cl2 = np.round(np.array(lambda_z_cl2), 3)
#print(lambda_z_cl1)
#print(lambda_z_cl2)


# latent_variables
z_cl1 = []
z_cl2 = []
for n in range(N):
    z_cl1.append(z_cl1_r[0][n])
    z_cl2.append(z_cl2_r[0][n])
z_cl1 = np.array(z_cl1)
z_cl2 = np.array(z_cl2)


# generated_data
generated_data_cl1_el1 = []
generated_data_cl1_el2 = []
generated_data_cl2_el1 = []
generated_data_cl2_el2 = []
for n in range(N):
    generated_data_cl1_el1.append(generated_data_cl1_el1_r[0][n])
    generated_data_cl1_el2.append(generated_data_cl1_el2_r[0][n])
    generated_data_cl2_el1.append(generated_data_cl2_el1_r[0][n])
    generated_data_cl2_el2.append(generated_data_cl2_el2_r[0][n])
generated_data_cl1_el1 = np.array(generated_data_cl1_el1)
generated_data_cl1_el2 = np.array(generated_data_cl1_el2)
generated_data_cl2_el1 = np.array(generated_data_cl2_el1)
generated_data_cl2_el2 = np.array(generated_data_cl2_el2)
#print(generated_data_cl1_el1)
#print(generated_data_cl1_el2)
#print(generated_data_cl2_el1)
#print(generated_data_cl2_el2)
temp_el1_1 = np.abs(z_cl1 * generated_data_cl1_el1)
temp_el1_2 = np.abs(z_cl2 * generated_data_cl2_el1)
#print(temp_el1_1)
#print(temp_el1_2)
generated_data_el1 = np.maximum(temp_el1_1, temp_el1_2)
#print(generated_data_el1)
temp_el2_1 = np.abs(z_cl1 * generated_data_cl1_el2)
temp_el2_2 = np.abs(z_cl2 * generated_data_cl2_el2)
#print(temp_el2_1)
#print(temp_el2_2)
generated_data_el2 = np.maximum(temp_el2_1, temp_el2_2)
#print(generated_data_el2)


# plot
ax = plt.axes()
lambda_z_cl1_cmax = np.max(lambda_z_cl1)
lambda_z_cl1_cmin = np.min(lambda_z_cl1)
lambda_z_cl2_cmax = np.max(lambda_z_cl2)
lambda_z_cl2_cmin = np.min(lambda_z_cl2)
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
ccm = cc.custumize_colormap(['mediumblue', 'orangered'])
plt.scatter(x=observation_el1, y=observation_el2, c=lambda_z_cl1, cmap=ccm, vmin=lambda_z_cl1_cmin, vmax=lambda_z_cl1_cmax, alpha=1.0, marker='x')
#gp.plot_datas(ax, observation, "r")
#plt.scatter(x=generated_data_el1, y=generated_data_el2, c=lambda_z_cl1, cmap=ccm, vmin=lambda_z_cl1_cmin, vmax=lambda_z_cl1_cmax, alpha=1.0, marker='*')
plt.colorbar()
plt.show()



