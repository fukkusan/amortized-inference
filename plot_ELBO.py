import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# constant
num_epochs = 100


# read csv
dataset_elbo = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.1.3_N100S10epoch100sample10/res_ELBO.csv")


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
