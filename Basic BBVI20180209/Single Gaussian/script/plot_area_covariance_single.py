import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os



num_epochs = 1000
num_samples = 100

#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver2.1/1000stepOK4/res_Area_covariance_ellipse_single_Gauss.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up/1000stepOK9/res_Area_covariance_ellipse_single_Gauss.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_exc_lambda_nu/1000stepOK3/res_Area_covariance_ellipse_single_Gauss.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up2/1000stepOK30/res_Area_covariance_ellipse_single_Gauss.csv")
dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up3/1000stepOK23/res_Area_covariance_ellipse_single_Gauss.csv")


area = []
area.append(dataset_area_covariance_ellipse['Area_ellipse_covariance'])
#print(area_cl1[0])


area_mean = []
start = 0
len_area = len(area[0])
while(start < len_area):
    area_mean.append(np.mean(area[0][start:start+num_samples-1]))
    start = start + num_samples
area_mean = np.array(area_mean)
print(area_mean)



plt.plot(range(num_epochs), area_mean, color='r')
plt.xlabel("epoch")
plt.ylabel("Areas")
plt.title("Areas_ellipse_covariance")
plt.show()

