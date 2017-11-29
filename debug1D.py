# Library
import numpy as np
import tensorflow as tf
import six
import numpy.random as nprand
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import math
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab
import Input_data as inp

# Projected Gradient Method
def PGMethod(vector_subspace, element_numbers):
    normal_vector = tf.ones(element_numbers)
    coefficient = tf.reduce_sum( tf.multiply(normal_vector, vector_subspace) )
    norm = tf.norm(normal_vector)
    oriented_vector = tf.multiply( coefficient, tf.divide(normal_vector, norm) )
    element_sum = tf.reduce_sum( tf.abs(oriented_vector) )
    vector_constrainted = tf.divide( oriented_vector, element_sum )
    
    return vector_constrainted

# Evidence Lower Bound
def ELBO(log_p, log_q):
    expectation_log_p_by_q = np.mean(log_p)
    expectation_log_q_by_q = np.mean(log_q)
    elbo = expectation_log_p_by_q - expectation_log_q_by_q
    
    return elbo

# Input data
N = 500  	# number of data points
K = 3    	# number of components
D = 1     	# dimensionality of data
S = 10		# sample
_alpha = 0.0
_beta = np.sqrt(0.1)
_gamma = 1.0

input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]

# Constants
num_epochs = 50
num_samples = 100

# learning rate
rho = 0.1

#***************************************
#* [Create Model]

print('***** Create Model *****')

# Constractor
# Parameters
# Input
X = tf.placeholder(tf.float32, shape = [N, D])

# Hyper parameters
alpha_mean = tf.constant(x_mean, shape=[D, K], dtype=tf.float32)
alpha_var = tf.constant(_beta, shape=[D, K], dtype=tf.float32)
gamma = tf.constant(_gamma, shape=[K], dtype=tf.float32)

# Variational parameters
print('* Variational parameters *')
lambda_pi = tf.Variable( tf.ones([K])*K, dtype=tf.float32, trainable=True, name='lambda_pi' )
#lambda_mu = tf.Variable( tf.truncated_normal([D, K], mean = 1.0, stddev=tf.sqrt(0.1) ), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_mu = tf.Variable( tf.ones([D, K]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32, trainable=True, name='lambda_z' )
print(lambda_pi)
print(lambda_mu)
print(lambda_z)

# Update count
update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )

# Update Distributions
# Variational approximation model
print('* Variational approximation model *')
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi, name='q_pi')
q_mu = tf.contrib.distributions.Normal(lambda_mu, tf.ones(K), name='q_mu')
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z, name='q_z')
print(q_pi)
print(q_mu)
print(q_z)

# Generative model
print('* Generative model *')
p_pi = tf.contrib.distributions.Dirichlet(gamma, name='p_pi' )
p_mu = tf.contrib.distributions.Normal(alpha_mean, alpha_var, name='p_mu' )
pi_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_pi.sample(sample_shape=[1])[0], lambda: p_pi.sample(sample_shape=[1])[0], name='pi_gene' )
mu_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_mu.sample(sample_shape=[1])[0], lambda: p_mu.sample(sample_shape=[1])[0], name='mu_gene' )
p_z = tf.contrib.distributions.OneHotCategorical( pi_gene, name='p_z'  )
generative_gauss = tf.contrib.distributions.Normal(mu_gene, tf.ones(K), name='generative_gauss' )
print(p_pi)
print(p_mu)
print(pi_gene)
print(mu_gene)
print(p_z)
print(generative_gauss)

# Inference variational parameters
# Sampling
print('* Sampling *')
with tf.name_scope('Sampling'):
    sample_gene_gauss = tf.Variable( tf.zeros([S, D, K]), name='sample_gene_gauss' )
    sample_gene_gauss = tf.assign( sample_gene_gauss, generative_gauss.sample(sample_shape=[S]), name='sample_gene_gauss' )
    sample_p_mu = tf.Variable( tf.zeros([S, D, K]), name='sample_p_mu' )
    sample_p_mu = tf.assign( sample_p_mu, p_mu.sample(sample_shape=[S]), name='sample_p_mu' )
    sample_p_z = tf.Variable( tf.zeros([S, K]), name='sample_p_z' )
    sample_p_z = tf.assign( sample_p_z, tf.to_float( p_z.sample(sample_shape=[S]) ), name='sample_p_z' )
    sample_p_pi = tf.Variable( tf.zeros([S, K]), name='sample_p_pi' )
    sample_p_pi = tf.assign( sample_p_pi, p_pi.sample(sample_shape=[S]), name='sample_p_pi' )
    sample_q_mu = tf.Variable( tf.zeros([S, D, K]), name='sample_q_mu' )
    sample_q_mu = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]), name='sample_q_mu')
    sample_q_z = tf.Variable( tf.zeros([S, N, K]), name='sample_q_z' )
    sample_q_z = tf.assign( sample_q_z, tf.to_float( q_z.sample(sample_shape=[S]) ), name='sample_q_z' )
    sample_q_pi = tf. Variable( tf.zeros([S, K]), name='sample_q_pi' )
    sample_q_pi = tf.assign( sample_q_pi, q_pi.sample(sample_shape=[S]), name='sample_q_pi' )
print(sample_gene_gauss)
print(sample_p_mu)
print(sample_p_z)
print(sample_p_pi)
print(sample_q_mu)
print(sample_q_z)
print(sample_q_pi)

# logarithmic distributions
print('* logarithmic distributions *')
with tf.name_scope('LogDistributions'):
    log_gene_gauss = generative_gauss.log_prob(sample_gene_gauss, name='log_gene_gauss')
    logpx = tf.reduce_sum( tf.multiply( tf.to_float( sample_p_z ), log_gene_gauss ), axis=1, name='logpx' )		#sample_shape=[N]
    log_p_x = tf.reduce_sum( logpx, axis=1, name='log_p_x' )
    log_p_mu = tf.reshape( tf.reduce_sum( p_mu.log_prob( sample_p_mu ), axis=2 ), shape=[S], name='log_p_mu' )
    log_p_pi = p_pi.log_prob( sample_p_pi, name='log_p_pi' )
    log_p_z = p_z.log_prob( sample_p_z, name='log_p_z' )
    log_dirichlet = q_pi.log_prob( sample_q_pi, name='log_dirichlet' )
    log_categorical = q_z.log_prob( sample_q_z, name='log_categorical' )
    log_gauss = q_mu.log_prob( sample_q_mu, name='log_gauss' )
    log_q_pi = log_dirichlet 
    log_q_mu = tf.reshape( tf.reduce_sum( log_gauss, axis=2 ), shape=[S], name='log_q_mu' )
    log_q_z = tf.reduce_sum( log_categorical, axis=1, name='log_q_z' )
    log_p = tf.add( tf.add( tf.add( log_p_x, log_p_z ), log_p_pi ), log_p_mu, name='log_p' ) 
    log_q = tf.add( tf.add( log_q_z, log_q_mu ), log_q_pi, name='log_q' ) 
    log_loss = tf.subtract( log_p, log_q, name='log_loss' )
print(log_gene_gauss)
print(logpx)
print(log_p_x)
print(log_p_mu)
print(log_p_pi)
print(log_p_z)
print(log_dirichlet)
print(log_categorical)
print(log_gauss)
print(log_q_mu)
print(log_q_z)
print(log_p)
print(log_q)
print(log_loss)

# Gradient
print('* Gradient *')
with tf.name_scope('Gradient'):
    grad_q_pi = []
    grad_q_mu = []
    grad_q_z = []
    for i in range(S):
        grad_q_pi.append( tf.gradients(log_q[i], lambda_pi) )
        grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
        grad_q_z.append( tf.gradients(log_q[i], lambda_z) )
    grad_q_pi = tf.convert_to_tensor(grad_q_pi, name='grad_q_pi')
    grad_q_mu = tf.convert_to_tensor(grad_q_mu, name='grad_q_mu')
    grad_q_z = tf.convert_to_tensor(grad_q_z, name='grad_q_z')
print(grad_q_pi)
print(grad_q_mu)
print(grad_q_z)

# Sample mean(Montecarlo Approximation)
print('* Sample Mean *')
with tf.name_scope('SampleMean'):
    element_wise_product_pi = []
    element_wise_product_mu = []
    element_wise_product_z = []
    for j in range(S):
        element_wise_product_pi.append( tf.multiply(grad_q_pi[j], log_loss[j]) )
        element_wise_product_mu.append( tf.multiply(grad_q_mu[j], log_loss[j]) )
        element_wise_product_z.append( tf.multiply(grad_q_z[j], log_loss[j]) )
    sample_mean_pi = tf.reduce_mean( element_wise_product_pi, axis = 0, name='sample_mean_pi' )[0]
    sample_mean_mu = tf.reduce_mean( element_wise_product_mu, axis = 0, name='sample_mean_mu' )[0]
    sample_mean_z = tf.reduce_mean( element_wise_product_z, axis = 0, name='sample_mean_z' )[0]
print(sample_mean_pi)
print(sample_mean_mu)
print(sample_mean_z)

# Update variational parameters
lambda_pi_new = tf.add(lambda_pi, tf.multiply(rho, sample_mean_pi), name='lambda_pi_new' )
lambda_mu_new = tf.add(lambda_mu, tf.multiply(rho, sample_mean_mu), name='lambda_mu_new' )
lambda_z_new = tf.add(lambda_z, tf.multiply(rho, sample_mean_z), name='lambda_z_new' )
print(lambda_pi_new)
print(lambda_mu_new)
print(lambda_z_new)

# Care Values
print('* Care Values *')
with tf.name_scope('Care'):
    _lambda_pi = []
    _lambda_pi.append( tf.split(lambda_pi_new, K, 0) )
    k=0
    while(k < K):
        _lambda_pi[0][k] = tf.cond( tf.less_equal( _lambda_pi[0][k][0], 0.0 ), lambda: tf.abs( tf.multiply(0.5, _lambda_pi[0][k]) ), lambda: _lambda_pi[0][k] )
        k = k + 1
    lambda_pi_care = tf.concat(_lambda_pi[0], 0, name='lambda_pi_care')
    
    _lambda_z = []
    _lambda_z.append( tf.split(lambda_z_new, N, 0) )
    n=0
    while(n < N):
        k=0
        while(k < K):
            #tf.less_equal( lambda_z[0][n][0][k], 0.0 )
            _lambda_z[0][n] = tf.cond( tf.less_equal( _lambda_z[0][n][0][k], 0.0 ), lambda: PGMethod(_lambda_z[0][n], [1, K]), lambda: _lambda_z[0][n] )
            k = k + 1
        _lambda_z[0][n] = tf.cond( tf.logical_or( tf.less_equal( tf.reduce_sum( _lambda_z[0][n] ), 0.9999999 ), tf.greater_equal( tf.reduce_sum( _lambda_z[0][n] ), 1.0000001 ) ), lambda: PGMethod( _lambda_z[0][n], [1, K] ), lambda: _lambda_z[0][n] )
        n = n + 1
    lambda_z_care = tf.concat(_lambda_z[0], 0, name='lambda_z_care')
print(lambda_pi_care)
print(lambda_z_care)

# Update 
print('* Update *')
lambda_pi_update = tf.assign(lambda_pi, lambda_pi_care, name='lambda_pi_update' )
lambda_mu_update = tf.assign(lambda_mu, lambda_mu_new, name='lambda_mu_update' )
lambda_z_update = tf.assign(lambda_z, lambda_z_care, name='lambda_z_update' )
print(lambda_pi_update)
print(lambda_mu_update)
print(lambda_z_update)

# Update time_step and learning parameter
update_counter = tf.assign( update_counter, tf.add(update_counter, tf.ones(1)) )
#rho = tf.minimum( 0.1, 1/tf.to_float(update_counter) )

# Values for plot (not use for Update)
sample_mu = q_mu.sample( sample_shape=[num_samples], name='sample_mu' )
sample_pi = q_pi.sample( sample_shape=[num_samples], name='sample_pi' )
sample_zq = q_z.sample( sample_shape=[num_samples], name='sample_zq' )
sample_zp = p_z.sample( sample_shape=[num_samples], name='sample_zp' )
print(sample_mu)
print(sample_pi)
print(sample_zq)
print(sample_zp)

#***************************************
#* [Run]

print('***** Session Init *****')

# Initiaize seesion
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('***** Run *****')

# Update
for epoch in range(num_epochs):
    result = sess.run([lambda_pi_update, lambda_mu_update, lambda_z_update, update_counter, sample_mu, sample_pi, sample_zq, sample_zp], feed_dict = {
        X: x
    })
    
    # for debug
    print(result[1])		# result[1] is lambda_mu

summary_writer = tf.summary.FileWriter('graph_BBVI', tf.get_default_graph())

#end
