import tensorflow as tf
import numpy as np
import pandas as pd
import Input_data_for_2_dim_GMM_K2 as inp
import matplotlib.pyplot as plt


# * Initialize *
print("* Initialize *")
# Constants
N = 1000  	# number of data points
K = 2    	# number of components
D = 2     	# dimensionality of data
S = 100		# sample
KS = K * S
#_alpha = 0.0
_beta = tf.constant(2.0, shape=[K, D, D])
_gamma = 2.5


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
unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([K, D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([K, D, D]), unit_matrices)
sample_unit = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([S, K, D, D])), name='sample_unit').to_dense()
hyper_alpha_mean = tf.constant([[3.2, 0.0], [-0.2, 3.2]], dtype=tf.float32, name='hyper_alpha_mean')
hyper_coe_alpha_var = tf.multiply(unit_matrices, _beta, name='hyper_coe_alpha_var')
hyper_gamma = tf.constant([0.3, 4.0], shape=[K], dtype=tf.float32, name='hyper_gamma')
hyper_V = [[[5.0, 0.0], [0.0, 5.0]], [[5.0, 0.0], [0.0, 5.0]]]
hyper_nu = tf.constant(2.05, shape=[K])


# Generative model
with tf.name_scope("GenerativeModel"):
    p_Lambda = tf.contrib.distributions.WishartFull(df=hyper_nu, scale=hyper_V)
    _covariance_p_mu = tf.multiply(hyper_coe_alpha_var, p_Lambda.sample(sample_shape=[1], seed=1))[0]
    covariance_p_mu = tf.matrix_inverse(_covariance_p_mu)
    p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean, covariance_matrix=covariance_p_mu)
    p_pi = tf.contrib.distributions.Dirichlet(hyper_gamma)
    #pred_counter = tf.reduce_sum(tf.to_float(tf.greater(update_counter, 0.0)))
    #pi_gene = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_pi_ass, lambda: p_pi.sample(sample_shape=[S]), name='pi_gene')  # pi for Cat(z|pi) is sampled by p_pi firstly and q_pi secondly
    pi_gene = p_pi.sample(sample_shape=[S])
    #mu_gene = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_mu_ass, lambda: p_mu.sample(sample_shape=[S]), name='mu_gene')  # mu for Normal(x|mu,Lambda) is sampled by p_mu firstly and q_mu secondly
    mu_gene = p_mu.sample(sample_shape=[S], seed=1)
    #covariance_generative_gauss = tf.cond(tf.equal(pred_counter, 1.0), lambda: tf.matrix_inverse(sample_q_Lambda_ass), lambda: tf.matrix_inverse(p_Lambda.sample(sample_shape=[S])))
    covariance_generative_gauss = tf.matrix_inverse(p_Lambda.sample(sample_shape=[S], seed=1))
    p_z = tf.contrib.distributions.OneHotCategorical(pi_gene)  # [S,K] sample_q_pi_ass
    generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu_gene, covariance_matrix=covariance_generative_gauss)    # [N, S, K, D]
    generative_data = tf.transpose(generative_gauss.sample([N]), perm=[1, 0, 2, 3])
    #generative_data = tf.transpose(generative_gauss.sample([N]), perm=[3, 1, 0, 2])
    latent_data = tf.transpose(p_z.sample([N]), perm=[1, 0, 2])
    #generative_x = tf.reduce_mean(tf.reduce_mean(generative_data, axis=0), axis=1)
    generative_x_cl1 = tf.transpose(tf.reduce_mean(generative_data, axis=0), perm=[1, 0, 2])[0]
    generative_x_cl2 = tf.transpose(tf.reduce_mean(generative_data, axis=0), perm=[1, 0, 2])[1]
    #generative_x = tf.tile(tf.slice(generative_x_cl1, [0, 0], [140, D]), tf.slice(generative_x_cl2, [0, 0], [60, D]))
    latent_z = tf.reduce_sum(tf.reduce_mean(tf.to_float(latent_data), axis=0), axis=0)
    #_generative_x = []
    #for s in range(S):
    #    for n in range(N):
    #        for k in range(K):
    #             _generative_x.append(tf.multiply(tf.to_float(latent_data[s][n][k]), generative_data[s][n][k]))
    #_generative_x = tf.multiply(tf.to_float(latent_data), generative_data[0])
    #generative_x = tf.reduce_mean(_generative_x, axis=0)



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


result = sess.run([latent_z, generative_x_cl1, generative_x_cl2])



fig = plt.figure()
for n in range(N):
    plt.scatter(result[1][n][0], result[1][n][1], color='b')
    plt.scatter(result[2][n][0], result[2][n][1], color='g')
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
inp.Input()
plt.show()

plt.bar([1, 2], [result[0][0], result[0][1]], width = 0.5, align='center')
plt.xticks([1, 2], ['class1', 'class2'])
plt.ylim([0, N])
plt.show()

print(result)





