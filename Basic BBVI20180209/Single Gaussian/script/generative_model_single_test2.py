import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.random as rand
import scipy.stats as stats
import Input_data_for_2_dim_Gauss as inp
import matplotlib.pyplot as plt
import calc_ELBO as cal_elbo
import Gaussian_plot2 as gp



# * Initialize *
print("* Initialize *")
# Constants
N = 1000  	# number of data points
D = 2     	# dimensionality of data
S = 100		# sample
_alpha = 0.0
_beta = tf.constant(10.0, shape=[D, D])
num_epochs = 1000
num_samples = S


## Observable
#input_data = inp.Input()
#x = input_data[0]
#x_mean = input_data[1]
#print("Input data")
#print(x)

## Input placeholder
#X = tf.placeholder(tf.float32, shape=[N, D], name='observable')


# observation
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_Lambda_up3/1000stepOK23/gauss_data.csv")
#dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3/1000stepOK15/gauss_data.csv")
dataset_observation = pd.read_csv("C:/work/Basic_BBVI_for_Single_Gaussian/csv/ver_lambda_mu_and_lambda_muLambda_3.1/1000stepOK4/gauss_data.csv")

observation_el1 = []
observation_el1.append( dataset_observation['element1'] )
observation_el2 = []
observation_el2.append( dataset_observation['element2'] )

observation = []
for n in range(len(observation_el1[0])):
    observation.append( [ observation_el1[0][n], observation_el2[0][n] ] )
observation = np.array(observation)


# Parameters
# Hyper parameters
unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([D, D]), unit_matrices)
sample_unit = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([S, D, D])), name='sample_unit').to_dense()
hyper_alpha_mean = tf.constant([3.1, 3.0], shape=[D], dtype=tf.float32, name='hyper_alpha_mean')
hyper_coe_alpha_var = tf.multiply(unit_matrices, _beta, name='hyper_coe_alpha_var')
hyper_V = tf.constant([[0.15, 0.0], [0.0, 0.1]])
hyper_nu = tf.constant(3.0)


# Generative model
with tf.name_scope("GenerativeModel"):
    p_Lambda = tf.contrib.distributions.WishartFull(df=hyper_nu, scale=hyper_V)
    sample_p_Lambda = tf.Variable(tf.ones([S, D, D]), name='sample_p_Lambda')
    _sample_p_Lambda = p_Lambda.sample(sample_shape=[S], seed=3)
    sample_p_Lambda_ass = tf.assign(sample_p_Lambda, _sample_p_Lambda)
    precision_p_mu = tf.multiply(hyper_coe_alpha_var, sample_p_Lambda_ass)[0]
    covariance_p_mu = tf.matrix_inverse(precision_p_mu)
    p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean, covariance_matrix=covariance_p_mu)
    mu_gene = p_mu.sample(sample_shape=[S])
    covariance_generative_gauss = tf.matrix_inverse(sample_p_Lambda_ass)
    generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu_gene, covariance_matrix=covariance_generative_gauss)    # [S, N, D]
    generative_data = tf.transpose(generative_gauss.sample([N]), perm=[1, 0, 2])
    generative_x = tf.reduce_mean(generative_data, axis=0)



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


result = sess.run(generative_x)

fig = plt.figure()
ax = plt.axes()
for n in range(N):
    plt.scatter(result[n][0], result[n][1], color='b')
plt.xlim([-4.0, 6.0])
plt.ylim([-4.0, 6.0])
gp.plot_datas(ax, observation, "r")
plt.show()

print(result)
