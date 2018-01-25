import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.random as rand
import scipy.stats as stats
import Input_data_for_2_dim_Gauss as inp
import matplotlib.pyplot as plt
import calc_ELBO as cal_elbo



# * Initialize *
print("* Initialize *")
# Constants
N = 1000  	# number of data points
D = 2     	# dimensionality of data
S = 100		# sample
_alpha = 0.0
_beta = tf.constant(12.0, shape=[D, D])
num_epochs = 1000
num_samples = 100
epsilon = tf.constant(0.0001)    # 10^{-4}
D_f = tf.to_float(D)
epsilon_plus = tf.add(1.0, epsilon)
epsilon_minus = tf.subtract(1.0, epsilon)
epsilon_pD = tf.add(D_f, epsilon)
epsilon_mD = tf.subtract(D_f, epsilon)


# Observable
input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]
print("Input data")
print(x)

# Input placeholder
X = tf.placeholder(tf.float32, shape=[N, D], name='observable')


# Parameters
# Hyper parameters
unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([D, D]), unit_matrices)
sample_unit = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([S, D, D])), name='sample_unit').to_dense()
hyper_alpha_mean = tf.constant([0.2, 0.0], shape=[D], dtype=tf.float32, name='hyper_alpha_mean')
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
for n in range(N):
    plt.scatter(result[n][0], result[n][1], color='b')
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
inp.Input()
plt.show()

print(result)
