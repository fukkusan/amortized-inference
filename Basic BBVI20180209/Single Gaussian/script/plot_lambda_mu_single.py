import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# constant
num_epochs = 1000


#dataset_eig_inv_exp_val_q_Lambda = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK27/res_eig_inv_expect_val_q_Lambda.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK27/gauss_data.csv")
#dataset_lambda_mu = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3/1000stepOK15/res_lambda_mu_single_Gauss.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3/1000stepOK15/gauss_data.csv")
dataset_lambda_mu = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3.1/1000stepOK4/res_lambda_mu_single_Gauss.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3.1/1000stepOK4/gauss_data.csv")



lambda_mu_el1_r = []
lambda_mu_el1_r.append(dataset_lambda_mu['lambda_mu_element1'])
#print(lambda_mu_el1_r)

lambda_mu_el2_r = []
lambda_mu_el2_r.append(dataset_lambda_mu['lambda_mu_element2'])
#print(lambda_mu_el2_r)

observation_el1 = []
observation_el1.append(dataset_observation['element1'])
#print(observation_el1)
observation_el2 = []
observation_el2.append(dataset_observation['element2'])
#print(observation_el2)


lambda_mu_el1 = []
for n in range(len(lambda_mu_el1_r[0])):
    lambda_mu_el1.append(lambda_mu_el1_r[0][n])
print(len(lambda_mu_el1))

lambda_mu_el2 = []
for n in range(len(lambda_mu_el2_r[0])):
    lambda_mu_el2.append(lambda_mu_el2_r[0][n])
print(len(lambda_mu_el2))


# observation
observation = []
for n in range(len(observation_el1[0])):
    observation.append([observation_el1[0][n], observation_el2[0][n]])
observation = np.array(observation)
#print(observation)


# unbiased estimator
mean_observation_el1 = np.mean(observation[0])
mean_observation_el2 = np.mean(observation[1])
#print(mean_observation_el1, mean_observation_el2)


# error
error1 = np.abs(lambda_mu_el1 - mean_observation_el1)/np.abs(mean_observation_el1)
error2 = np.abs(lambda_mu_el2 - mean_observation_el2)/np.abs(mean_observation_el2)
data_frame_error = pd.DataFrame(index=[], columns=['error1', 'error2'])
for n in range(num_epochs):
    series = pd.Series([error1[n], error2[n]], index=data_frame_error.columns)
    data_frame_error = data_frame_error.append(series, ignore_index=True)


# plot
fig = plt.figure()
ax = plt.axes()
plt.subplot(2, 1, 1)
plt.plot(lambda_mu_el1, color='cyan')
plt.plot(mean_observation_el1*np.ones(shape=[num_epochs]), color='midnightblue')
plt.ylabel("lambda_mu_element1")
plt.subplot(2, 1, 2)
plt.plot(lambda_mu_el2, color='lightsalmon')
plt.plot(mean_observation_el2*np.ones(shape=[num_epochs]), color='tomato')
plt.ylabel("lambda_mu_element2")

plt.xlabel("epoch")
plt.suptitle("lambda_mu")

plt.show()


plt.subplot(2, 1, 1)
plt.plot(error1, color='forestgreen')
plt.ylabel("error1")
plt.subplot(2, 1, 2)
plt.plot(error2, color='lime')
plt.ylabel("error2")

plt.xlabel("epoch")
plt.suptitle("difference lambda_mu - mean_observation")

plt.show()

data_frame_error.to_csv("res_difference_lambda_mu - mean_observation.csv", index=False)




