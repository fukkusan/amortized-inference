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
#import os
#import io
#import time
import gc
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
S = 100		# sample
_alpha = 0.0
_beta = np.sqrt(0.1)
_gamma = 1.0

input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]



# Constractor
# Parameters
# Input
X = tf.placeholder(tf.float32, shape = [N, D])

# Hyper parameters
alpha_mean = tf.constant(x_mean, shape=[D, K], dtype=tf.float32)
alpha_var = tf.constant(_beta, shape=[D, K], dtype=tf.float32)
gamma = tf.constant(_gamma, shape=[K], dtype=tf.float32)

# Variational parameters
lambda_pi = tf.Variable( tf.ones([K])*K, dtype=tf.float32, trainable=True, name='lambda_pi' )
#lambda_mu = tf.Variable( tf.truncated_normal([D, K], mean = 1.0, stddev=tf.sqrt(0.1) ), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_mu = tf.Variable( tf.ones([D, K]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32, trainable=True, name='lambda_z' )

# Update count
update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )

# Save previous variational parameters
prev_lambda_pi = tf.Variable( tf.ones(K)/K, dtype=tf.float32 )
#prev_lambda_mu = tf.Variable( tf.truncated_normal( [D, K], mean = 1.0, stddev=tf.sqrt(0.1) ), dtype=tf.float32 )
prev_lambda_mu = tf.Variable( tf.ones( [D, K] ), dtype=tf.float32 )
prev_lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32 )

# Constants
num_epochs = 1001
num_samples = 100

# learning rate
rho = 0.1


# initialize distributions
# Variational approximation model
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi)				
q_mu = tf.contrib.distributions.Normal(lambda_mu, tf.ones(K))		
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z)			

# Generative model
p_pi = tf.contrib.distributions.Dirichlet(gamma)					
p_mu = tf.contrib.distributions.Normal(alpha_mean, alpha_var)		
pi_gene = p_pi.sample(sample_shape=[1])[0]
mu_gene = p_mu.sample(sample_shape=[1])[0]
p_z = tf.contrib.distributions.OneHotCategorical( pi_gene )			
generative_gauss = tf.contrib.distributions.Normal(mu_gene, tf.ones(K))


# Inference variational parameters
#Sampling
sample_gene_gauss = tf.Variable( tf.zeros([S, D, K]) )
sample_gene_gauss = tf.assign( sample_gene_gauss, generative_gauss.sample(sample_shape=[S]) )
sample_p_mu = tf.Variable( tf.zeros([S, D, K]) )
sample_p_mu = tf.assign( sample_p_mu, p_mu.sample(sample_shape=[S]) )
sample_p_z = tf.Variable( tf.zeros([S, K]) )
sample_p_z = tf.assign( sample_p_z, tf.to_float( p_z.sample(sample_shape=[S]) ) )
sample_p_pi = tf.Variable( tf.zeros([S, K]) )
sample_p_pi = tf.assign( sample_p_pi, p_pi.sample(sample_shape=[S]) )
sample_q_mu = tf.Variable( tf.zeros([S, D, K]) )
sample_q_mu = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )
sample_q_z = tf.Variable( tf.zeros([S, N, K]) )
sample_q_z = tf.assign( sample_q_z, tf.to_float( q_z.sample(sample_shape=[S]) ) )
sample_q_pi = tf. Variable( tf.zeros([S, K]) )
sample_q_pi = tf.assign( sample_q_pi, q_pi.sample(sample_shape=[S]) )


# logarithmic distributions
log_gene_gauss = generative_gauss.log_prob(sample_gene_gauss)
logpx = tf.reduce_sum( tf.multiply( tf.to_float( sample_p_z ), log_gene_gauss ), axis=1 )					#sample_shape=[N]
log_p_x = tf.reduce_sum( logpx, axis=1 )
log_p_mu = tf.reshape( tf.reduce_sum( p_mu.log_prob( sample_p_mu ), axis=2 ), shape=[S] )
log_p_pi = p_pi.log_prob( sample_p_pi )
log_p_z = p_z.log_prob( sample_p_z )
log_dirichlet = q_pi.log_prob( sample_q_pi )
log_categorical = q_z.log_prob( sample_q_z )
log_gauss = q_mu.log_prob( sample_q_mu )
log_q_pi = log_dirichlet 
log_q_mu = tf.reshape( tf.reduce_sum( log_gauss, axis=2 ), shape=[S] )
log_q_z = tf.reduce_sum( log_categorical, axis=1 )

log_p = tf.add( tf.add( tf.add( log_p_x, log_p_z ), log_p_pi ), log_p_mu ) 
log_q = tf.add( tf.add( log_q_z, log_q_mu ), log_q_pi ) 
log_loss = tf.subtract( log_p, log_q )


# Gradient
grad_q_pi = []
grad_q_mu = []
grad_q_z = []
for i in range(S):
    grad_q_pi.append( tf.gradients(log_q[i], lambda_pi) )
    grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
    grad_q_z.append( tf.gradients(log_q[i], lambda_z) )
grad_q_pi = tf.convert_to_tensor(grad_q_pi)
grad_q_mu = tf.convert_to_tensor(grad_q_mu)
grad_q_z = tf.convert_to_tensor(grad_q_z)
test= tf.gradients(q_mu.log_prob( sample_q_mu ), lambda_mu)

# Sample mean(Montecarlo Approximation)
element_wise_product_pi = []
element_wise_product_mu = []
element_wise_product_z = []
for j in range(S):
    element_wise_product_pi.append( tf.multiply(grad_q_pi[j], log_loss[j]) )
    element_wise_product_mu.append( tf.multiply(grad_q_mu[j], log_loss[j]) )
    element_wise_product_z.append( tf.multiply(grad_q_z[j], log_loss[j]) )
sample_mean_pi = tf.reduce_mean( element_wise_product_pi, axis = 0 )[0]
sample_mean_mu = tf.reduce_mean( element_wise_product_mu, axis = 0 )[0]
sample_mean_z = tf.reduce_mean( element_wise_product_z, axis = 0 )[0]


# Update variational parameters
lambda_pi = tf.assign(lambda_pi, tf.add(lambda_pi, tf.multiply(rho, sample_mean_pi)) )
lambda_mu = tf.assign(lambda_mu, tf.add(lambda_mu, tf.multiply(rho, sample_mean_mu)) )
lambda_z = tf.assign(lambda_z, tf.add(lambda_z, tf.multiply(rho, sample_mean_z)) )



# Care Values
_lambda_pi = []
_lambda_pi.append( tf.split(lambda_pi, K, 0) )
k=0
while(k < K):
    _lambda_pi[0][k] = tf.cond( tf.less_equal( _lambda_pi[0][k][0], 0.0 ), lambda: tf.abs( tf.multiply(0.5, _lambda_pi[0][k]) ), lambda: _lambda_pi[0][k] )
    k = k + 1
if(k == K):
    lambda_pi = tf.concat(_lambda_pi[0], 0)
del _lambda_pi[:]
gc.collect()


_lambda_z = []
_lambda_z.append( tf.split(lambda_z, N, 0) )
n=0
while(n < N):
    k=0
    while(k < K):
        #tf.less_equal( lambda_z[0][n][0][k], 0.0 )
        _lambda_z[0][n] = tf.cond( tf.less_equal( _lambda_z[0][n][0][k], 0.0 ), lambda: PGMethod(_lambda_z[0][n], [1, K]), lambda: _lambda_z[0][n] )
        k = k + 1
    _lambda_z[0][n] = tf.cond( tf.logical_or( tf.less_equal( tf.reduce_sum( _lambda_z[0][n] ), 0.9999999 ), tf.greater_equal( tf.reduce_sum( _lambda_z[0][n] ), 1.0000001 ) ), lambda: PGMethod( _lambda_z[0][n], [1, K] ), lambda: _lambda_z[0][n] )
    n = n + 1
if(n == N):
    lambda_z = tf.concat(_lambda_z[0], 0)
del _lambda_z[:]
gc.collect()


# Deal with nan and inf
for i in range(K):
    inf_pi_msg = tf.cond( tf.equal( tf.is_inf(lambda_pi)[i], True ), lambda: True, lambda: False )
    if inf_pi_msg == True:
        print("lambda_pi is inf")
    lambda_pi = tf.cond( tf.equal( tf.is_inf(lambda_pi)[i], True ), lambda: prev_lambda_pi, lambda: lambda_pi )
    nan_pi_msg = tf.cond( tf.equal( tf.is_nan(lambda_pi)[i], True ), lambda: True, lambda: False )
    if nan_pi_msg == True:
        print("lambda_pi is nan")
    lambda_pi = tf.cond( tf.equal( tf.is_nan(lambda_pi)[i], True ), lambda: prev_lambda_pi, lambda: lambda_pi ) 
    inf_mu_msg = tf.cond( tf.equal( tf.is_inf(lambda_mu)[0][i], True ), lambda: True, lambda: False )
    if inf_mu_msg == True:
        print("lambda_mu is inf")
    lambda_mu = tf.cond( tf.equal( tf.is_inf(lambda_mu)[0][i], True ), lambda: prev_lambda_mu, lambda: lambda_mu ) 
    nan_mu_msg = tf.cond( tf.equal( tf.is_nan(lambda_mu)[0][i], True ), lambda: True, lambda: False )
    if nan_mu_msg == True:
        print("lambda_mu is nan")
    lambda_mu = tf.cond( tf.equal( tf.is_nan(lambda_mu)[0][i], True ), lambda: prev_lambda_mu, lambda: lambda_mu ) 
for j in range(N):
    for m in range(K):
        inf_z_msg = tf.cond( tf.equal( tf.is_inf(lambda_z[j][m]), True ), lambda: True, lambda: False )
        if inf_z_msg == True:
            print("lambda_z is inf")
        lambda_z = tf.cond( tf.equal( tf.is_inf(lambda_z[j][m]), True ), lambda: prev_lambda_z, lambda: lambda_z ) 
        nan_z_msg = tf.cond( tf.equal( tf.is_nan(lambda_z[j][m]), True ), lambda: True, lambda: False )
        if nan_z_msg == True:
            print("lambda_z is nan")
        lambda_z = tf.cond( tf.equal( tf.is_nan(lambda_z[j][m]), True ), lambda: prev_lambda_z, lambda: lambda_z ) 

# Previous lambda
prev_lambda_pi = tf.assign( prev_lambda_pi, lambda_pi )
prev_lambda_mu = tf.assign( prev_lambda_mu, lambda_mu )
prev_lambda_z = tf.assign( prev_lambda_z, lambda_z )


# Update Distributions
# Variational approximation model
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi)				
q_mu = tf.contrib.distributions.Normal(lambda_mu, tf.ones(K))		
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z)			

# Generative model
p_pi = tf.contrib.distributions.Dirichlet(gamma)					
p_mu = tf.contrib.distributions.Normal(alpha_mean, alpha_var)		
pi_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_pi.sample(sample_shape=[1])[0], lambda: p_pi.sample(sample_shape=[1])[0] )
mu_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_mu.sample(sample_shape=[1])[0], lambda: p_mu.sample(sample_shape=[1])[0] )
p_z = tf.contrib.distributions.OneHotCategorical( pi_gene )			
generative_gauss = tf.contrib.distributions.Normal(mu_gene, tf.ones(K))


# Update time_step and learning parameter
update_counter = tf.assign( update_counter, tf.add(update_counter, tf.ones(1)) )
rho = tf.minimum( 0.1, 1/tf.to_float(update_counter) )



# Values for plot
vps_mu_0 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_mu_1 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_mu_2 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_pi_0 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_pi_1 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_pi_2 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_z_0 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_z_1 = np.array(np.zeros(num_epochs), dtype=np.float32)
vps_z_2 = np.array(np.zeros(num_epochs), dtype=np.float32)
log_likelihood_p = np.array(np.zeros(num_epochs), dtype=np.float32)
list_log_q = np.array(np.zeros(num_epochs), dtype=np.float32)
elbo = np.array(np.zeros(num_epochs), dtype=np.float32)
df = pd.DataFrame(index=[], columns=['class1', 'class2', 'class3'])
count = 1
sample_mu = q_mu.sample( sample_shape=[num_samples] )		# sampling 100 mu
sample_pi = q_pi.sample( sample_shape=[num_samples] )		# sampling 100 pi
sample_zq = q_z.sample( sample_shape=[num_samples] )		# sampling 100 z from q(z)
sample_zp = p_z.sample( sample_shape=[num_samples] )		# sampling 100 z from p(z)


# Initiaize seesion
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# Update
for epoch in range(num_epochs):
    result = sess.run([lambda_pi, lambda_mu, lambda_z, sample_mu, sample_pi, sample_zq, sample_zp, log_p, log_q, log_gene_gauss], feed_dict = {
        X: x
    })
    
    
    # for debug
    print(result[1])		# result[1] is lambda_mu
    #print(result[9])
    #print(result[3])
    mu_0 = np.array(np.zeros(num_samples), dtype=np.float32)
    mu_1 = np.array(np.zeros(num_samples), dtype=np.float32)
    mu_2 = np.array(np.zeros(num_samples), dtype=np.float32)
    pi_0 = np.array(np.ones(num_samples), dtype=np.float32)
    pi_1 = np.array(np.ones(num_samples), dtype=np.float32)
    pi_2 = np.array(np.ones(num_samples), dtype=np.float32)
    zq_0 = 0
    zq_1 = 0
    zq_2 = 0
    zp_0 = 0
    zp_1 = 0
    zp_2 = 0
    for i in range(num_samples):
        mu_0[i] = result[3][i][0][0]
        mu_1[i] = result[3][i][0][1]
        mu_2[i] = result[3][i][0][2]
        pi_0[i] = result[4][i][0]
        pi_1[i] = result[4][i][1]
        pi_2[i] = result[4][i][2]
        zq_0 = zq_0 + result[5][0][i][0]
        zq_1 = zq_1 + result[5][0][i][1]
        zq_2 = zq_2 + result[5][0][i][2]
        zp_0 = zp_0 + result[6][i][0]
        zp_1 = zp_1 + result[6][i][1]
        zp_2 = zp_2 + result[6][i][2]
    if epoch%100 == 0:
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot([-1.0, -1.0, -1.0], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="b")
        plt.plot([0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="g")
        plt.plot([1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="r")
        plt.vlines(x=mu_0, ymin=-1, ymax=1, color="c", linestyle="dotted", label="mu_0")
        plt.vlines(x=mu_1, ymin=-1, ymax=1, color="y", linestyle="dotted", label="mu_1")
        plt.vlines(x=mu_2, ymin=-1, ymax=1, color="m", linestyle="dotted", label="mu_2")
        plt.scatter(x, np.zeros(np.size(x)), color="k", label="x_train", marker="x")
        plt.ylim([0, 1])
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot([0.7, 0.7, 0.7], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="b")
        plt.plot([0.2, 0.2, 0.2], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="g")
        plt.plot([0.1, 0.1, 0.1], [-1.0, 0.0, 1.0], linewidth=2, linestyle="solid", color="r")
        plt.vlines(x=pi_0, ymin=-1, ymax=1, color="c", linestyle="dotted", label="pi_0")
        plt.vlines(x=pi_1, ymin=-1, ymax=1, color="y", linestyle="dotted", label="pi_1")
        plt.vlines(x=pi_2, ymin=-1, ymax=1, color="m", linestyle="dotted", label="pi_2")
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.scatter([1, 2, 3], [zq_0, zq_1, zq_2], label="z sampled q_z")		# [1,2,3] is numbers for each classes
        plt.ylim([0, 100])
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.scatter([1, 2, 3], [zp_0, zp_1, zp_2], label="z sampled p_z")
        plt.ylim([0, 100])
        plt.legend()
        plt.show()
        plt.clf()
        plt.close()
    vps_mu_0[epoch] = result[1][0][0]
    vps_mu_1[epoch] = result[1][0][1]
    vps_mu_2[epoch] = result[1][0][2]
    vps_pi_0[epoch] = result[0][0]
    vps_pi_1[epoch] = result[0][1]
    vps_pi_2[epoch] = result[0][2]
    vps_z_0[epoch] = result[2][0][0]		# 2nd index is data point
    vps_z_1[epoch] = result[2][0][1]
    vps_z_2[epoch] = result[2][0][2]
    log_likelihood_p[epoch] = np.mean(result[7])
    list_log_q[epoch] = np.mean(result[8])
    elbo[epoch] = ELBO(result[7], result[8])
    se = pd.Series([vps_mu_0[epoch], vps_mu_1[epoch], vps_mu_2[epoch]], index=df.columns)
    df = df.append(se, ignore_index=True)
    print(count)
    count = count + 1
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.scatter(range(num_epochs), vps_mu_0, color="b", label="lambda_mu0")
plt.scatter(range(num_epochs), vps_mu_1, color="g", label="lambda_mu1")
plt.scatter(range(num_epochs), vps_mu_2, color="r", label="lambda_mu2")
plt.xlim([0, num_epochs])
#plt.ylim([0,1])
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(range(num_epochs), vps_pi_0, color="b", label="lambda_pi0")
plt.scatter(range(num_epochs), vps_pi_1, color="g", label="lambda_pi1")
plt.scatter(range(num_epochs), vps_pi_2, color="r", label="lambda_pi2")
plt.legend()
plt.xlim([0, num_epochs])
plt.show()
plt.subplot(3, 1, 1)
plt.scatter(range(num_epochs), vps_z_0, label="lambda_z10")
plt.xlim([0, num_epochs])
plt.legend()
plt.subplot(3, 1, 2)
plt.scatter(range(num_epochs), vps_z_1, label="lambda_z11")
plt.xlim([0, num_epochs])
plt.legend()
plt.subplot(3, 1, 3)
plt.scatter(range(num_epochs), vps_z_2, label="lambda_z12")
plt.xlim([0, num_epochs])
plt.legend()
plt.show()
plt.clf()
plt.close()
plt.plot(range(num_epochs), log_likelihood_p, label="log_p")
plt.plot(range(num_epochs), list_log_q, label="log_q")
plt.title("log_p and log_q")
plt.legend()
plt.show()
plt.plot(range(num_epochs), elbo, label="ELBO")
plt.title("ELBO")
plt.legend()
plt.show()
plt.clf()
plt.close()
df.to_csv("output_lambda_mu.csv", index=False)
    
    
    


