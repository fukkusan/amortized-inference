import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os



num_epochs = 1000
num_samples = 100

#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_N1000S100epoch1000sample100/1000stepOK3/res_Area_covariance_ellipse.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.6_ho_N1000S100epoch1000sample100/1000stepOK2/res_Area_covariance_ellipse.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S100epoch1000sample100/1000stepOK13/res_Area_covariance_ellipse.csv")
dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.7_N1000S200epoch1000sample200/1000stepOK/res_Area_covariance_ellipse.csv")
#dataset_area_covariance_ellipse = pd.read_csv("C:/work/Basic_BBVI_for_2-dim_GMM/csv/ver2.2.5_N200S100epoch1000sample100/1000stepOK2/res_Area_covariance_ellipse.csv")


area_cl1 = []
area_cl1.append(dataset_area_covariance_ellipse['Area_covariance_ellipse_class1'])
#print(area_cl1[0])
area_cl2 = []
area_cl2.append(dataset_area_covariance_ellipse['Area_covariance_ellipse_class2'])


#area_mean_cl1 = []
#start_1 = 0
#len_area_cl1 = len(area_cl1[0])
#while(start_1 < len_area_cl1):
#    area_mean_cl1.append(np.mean(area_cl1[0][start_1:start_1+num_samples-1]))
#    start_1 = start_1 + num_samples
#area_mean_cl1 = np.array(area_mean_cl1)
#print(area_mean_cl1)
#
#area_mean_cl2 = []
#start_2 = 0
#len_area_cl2 = len(area_cl2[0])
#while(start_2 < len_area_cl2):
#    area_mean_cl2.append(np.mean(area_cl2[0][start_2:start_2+num_samples-1]))
#    start_2 = start_2 + num_samples
#area_mean_cl2 = np.array(area_mean_cl2)
#print(area_mean_cl2)


plt.plot(range(num_epochs), area_cl1[0], color='r')
plt.plot(range(num_epochs), area_cl2[0], color='b')
plt.xlabel("epoch")
plt.ylabel("Areas")
plt.title("Areas_ellipse_covariance")
plt.show()

