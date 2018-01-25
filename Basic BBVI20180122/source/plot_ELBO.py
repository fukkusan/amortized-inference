import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# constant
num_epochs = 1000


# read csv
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_ELBO.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.4_N1000S100epoch1000sample100/1000stepOK/res_ELBO.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_N1000S100epoch1000sample100/1000stepOK3/res_ELBO_K2.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/res_ELBO_K2.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/no_care_N200S100epoch1000sample100/res_ELBO_K2_nocare.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver2.1/1000stepOK4/res_ELBO_single_Gauss.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up/1000stepOK9/res_ELBO_single_Gauss.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_exc_lambda_nu/1000stepOK/res_ELBO_single_Gauss.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda/1000stepOK28/res_ELBO_single_Gauss.csv")
#dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_2/1000stepOK21/res_ELBO_single_Gauss.csv")
dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK27/res_ELBO_single_Gauss.csv")


log_p_mean_r = []
log_p_mean_r.append(dataset_elbo['log_p_mean'])
#print(log_p_mean_r)

log_q_mean_r = []
log_q_mean_r.append(dataset_elbo['log_q_mean'])
#print(log_q_mean_r)

elbo_r = []
elbo_r.append(dataset_elbo['ELBO'])
print(elbo_r)

log_p_mean = []
log_q_mean = []
elbo = []
for step in range(num_epochs):
    elbo.append(elbo_r[0][step])
    log_p_mean.append(log_p_mean_r[0][step])
    log_q_mean.append(log_q_mean_r[0][step])


# plot
fig = plt.figure()
ax = plt.axes()
plt.plot(range(num_epochs), elbo)
plt.xlabel("epoch")
plt.ylabel("ELBO")
plt.title("ELBO")
plt.show()
