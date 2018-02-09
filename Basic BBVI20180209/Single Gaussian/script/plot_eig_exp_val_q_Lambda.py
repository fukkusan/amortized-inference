import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_epochs = 1000


#dataset_eig_inv_exp_val_q_Lambda = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK27/res_eig_inv_expect_val_q_Lambda.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK27/gauss_data.csv")
dataset_eig_inv_exp_val_q_Lambda = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up3/1000stepOK23/res_eig_inv_expect_val_q_Lambda.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up3/1000stepOK23/gauss_data.csv")


eig_inv_exp_val_1 = []
eig_inv_exp_val_1.append(dataset_eig_inv_exp_val_q_Lambda['eig_inv_expect_val_q_Lambda_eig1'])
#print(eig_inv_exp_val_1)
eig_inv_exp_val_2 = []
eig_inv_exp_val_2.append(dataset_eig_inv_exp_val_q_Lambda['eig_inv_expect_val_q_Lambda_eig2'])
#print(eig_inv_exp_val_2)

observation_el1 = []
observation_el1.append(dataset_observation['element1'])
#print(observation_el1)
observation_el2 = []
observation_el2.append(dataset_observation['element2'])
#print(observation_el2)


# eig_inv_exp_val
eig_inv_exp_val_el1 = []
for n in range(len(eig_inv_exp_val_1[0])):
    eig_inv_exp_val_el1.append(eig_inv_exp_val_1[0][n])
eig_inv_exp_val_el1 = np.array(eig_inv_exp_val_el1)
#print(eig_inv_exp_val_el1)

eig_inv_exp_val_el2 = []
for n in range(len(eig_inv_exp_val_2[0])):
    eig_inv_exp_val_el2.append(eig_inv_exp_val_2[0][n])
eig_inv_exp_val_el2 = np.array(eig_inv_exp_val_el2)
#print(eig_inv_exp_val_el2)


# observation
observation = []
for n in range(len(observation_el1[0])):
    observation.append([observation_el1[0][n], observation_el2[0][n]])
observation = np.array(observation)
#print(observation)


# unbiased estimator
unbiasd_estimator_cov = np.cov(observation.T)
print(unbiasd_estimator_cov)
eig_val_unbiasd_estimator_cov = np.linalg.eig(unbiasd_estimator_cov)[0]
print(eig_val_unbiasd_estimator_cov)


# error
error1 = np.abs(eig_inv_exp_val_el1 - eig_val_unbiasd_estimator_cov[0])/eig_val_unbiasd_estimator_cov[0]
error2 = np.abs(eig_inv_exp_val_el2 - eig_val_unbiasd_estimator_cov[1])/eig_val_unbiasd_estimator_cov[1]
#print(error1)
#print(error2)
data_frame_error = pd.DataFrame(index=[], columns=['error1', 'error2'])
for n in range(num_epochs):
    series = pd.Series([error1[n], error2[n]], index=data_frame_error.columns)
    data_frame_error = data_frame_error.append(series, ignore_index=True)


# plot
fig = plt.figure()
ax = plt.axes()
plt.subplot(2, 1, 1)
plt.plot(eig_inv_exp_val_el1, color='cyan')
plt.plot(eig_val_unbiasd_estimator_cov[0]*np.ones(shape=num_epochs), color='midnightblue')
plt.ylabel("eig_inv_exp_val_1")
plt.subplot(2, 1, 2)
plt.plot(eig_inv_exp_val_el2, color='lightsalmon')
plt.plot(eig_val_unbiasd_estimator_cov[1]*np.ones(shape=num_epochs), color='tomato')
plt.ylabel("eig_inv_exp_val_2")

plt.xlabel("epoch")
plt.suptitle("eig_inv_expect_val_q_Lambda")
    
plt.show()


plt.subplot(2, 1, 1)
plt.plot(error1, color='forestgreen')
plt.ylabel("error1")
plt.subplot(2, 1, 2)
plt.plot(error2, color='lime')
plt.ylabel("error2")

plt.xlabel("epoch")
plt.suptitle("difference eig_inv_exp_val - eig_val_unbiasd_estimator_cov")
plt.show()

data_frame_error.to_csv("res_difference_eig_inv_exp_val-eig_val_unbiasd_estimator_cov.csv", index=False)


