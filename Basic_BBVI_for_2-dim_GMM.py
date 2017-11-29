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
import os
import io
import time
import gc
from tensorflow.python import debug as tf_debug
import Input_data_2dGMM as inp
from scipy import linalg




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




#***************************************** 
# Constant and Observable
N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample
_alpha = 0.0
_beta = np.sqrt(0.1)
_gamma = 1.0

input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]




#***************************************
#* [Create Model]

print('***** Create Model *****')
# Initialize
# Parameters
# Input
X = tf.placeholder(tf.float32, shape = [N, D])

# Hyper parameters
hyper_alpha_mean = tf.constant(x_mean, shape=[K, D], dtype=tf.float32)
hyper_coe_alpha_var = tf.constant(_beta, shape=[1], dtype=tf.float32)
hyper_gamma = tf.constant(_gamma, shape=[K], dtype=tf.float32)
#blocked_hyper_V_1 = tf.diag( tf.ones([1,D]) )[0]
#blocked_hyper_V_2 = tf.diag( tf.ones([1,D]) )[0]
#blocked_hyper_V_3 = tf.diag( tf.ones([1,D]) )[0]
unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
#hyper_V = tf.reshape( tf.concat( [tf.concat( [blocked_hyper_V_1, blocked_hyper_V_2], 0 ), blocked_hyper_V_3], 0 ), [K, D, D] )
hyper_V = tf.reshape( unit_matrix, [K, D, D] )
hyper_nu = tf.constant(2.0, shape=[K])

# Variational parameters
lambda_pi = tf.Variable( tf.ones([K])*K, dtype=tf.float32, trainable=True, name='lambda_pi' )
#lambda_mu = tf.Variable( tf.truncated_normal([D, K], mean = 1.0, stddev=tf.sqrt(0.1) ), dtype=tf.float32, trainable=True, name='lambda_mu' )
#lambda_mu = tf.Variable( tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_mu = tf.Variable( tf.ones([D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
#block_lambda_muLambda_1 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_muLambda_2 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_muLambda_3 = tf.diag( tf.ones([1,D]) )[0]
#lambda_muLambda = tf.Variable( tf.reshape( tf.concat( [tf.concat( [block_lambda_muLambda_1, block_lambda_muLambda_2], 0 ), block_lambda_muLambda_3], 0 ), [K, D, D] ), dtype=tf.float32, trainable=True, name='lambda_muLambda' )
lambda_muLambda = tf.Variable( tf.ones([K]), dtype=tf.float32, trainable=True, name='lambda_muLambda' )
lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32, trainable=True, name='lambda_z' )
#block_lambda_Lambda_1 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_Lambda_2 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_Lambda_3 = tf.diag( tf.ones([1,D]) )[0]
#unit_matrix = tf.reshape( tf.concat( [tf.concat( [block_lambda_Lambda_1, block_lambda_Lambda_2], 0 ), block_lambda_Lambda_3], 0 ), [K, D, D] )
lambda_Lambda = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )

# Update count
update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )

# Save previous variational parameters
prev_lambda_pi = tf.Variable( tf.ones(K)*K, dtype=tf.float32 )
#prev_lambda_mu = tf.Variable( tf.truncated_normal( [D, K], mean = 1.0, stddev=tf.sqrt(0.1) ), dtype=tf.float32 )
prev_lambda_mu = tf.Variable( tf.ones([D, K]), dtype=tf.float32, trainable=True )
prev_lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32 )
#prev_lambda_Lambda = tf.Variable( tf.concat( [tf.concat( [block_lambda_Lambda_1, block_lambda_Lambda_2], 0 ), block_lambda_Lambda_3], 0 ), dtype=tf.float32, trainable=True )
prev_lambda_Lambda = tf.Variable( unit_matrix, dtype=tf.float32, trainable=True )
prev_lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True )

# Constants
num_epochs = 10
num_samples = 100

# learning rate
rho = tf.Variable( tf.constant(0.1, shape=[1]), dtype=tf.float32, trainable=True )


# Initialize and update distributions
# Variational approximation model
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z)			
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi)				
#q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(lambda_mu, lambda_muLambda)
precision = []
for k in range(K):
    precision.append( tf.multiply( lambda_muLambda[k], unit_matrix[k] ) )
q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( lambda_mu, tf.matrix_inverse( precision ) )
q_Lambda = tf.contrib.distributions.WishartFull( df=lambda_nu, scale=lambda_Lambda )

# Generative model
p_Lambda = tf.contrib.distributions.WishartFull( df=hyper_nu, scale=hyper_V )
p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( hyper_alpha_mean, tf.multiply( hyper_coe_alpha_var, tf.matrix_inverse( p_Lambda.sample( sample_shape=[1] )[0] ) ) )		
p_pi = tf.contrib.distributions.Dirichlet(hyper_gamma)					
pi_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_pi.sample(sample_shape=[N]), lambda: p_pi.sample(sample_shape=[N]), name='pi_gene' )
mu_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_mu.sample(sample_shape=[N]), lambda: p_mu.sample(sample_shape=[N]), name='mu_gene' )
p_z = tf.contrib.distributions.OneHotCategorical( pi_gene )			
generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance( mu_gene, tf.matrix_inverse( p_Lambda.sample( sample_shape=[N] ) ) )
#generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance( mu_gene, tf.matrix_inverse( p_Lambda.sample( sample_shape=[1] ) ) )



# Inference variational parameters
# Sampling
sample_gene_gauss = tf.Variable( tf.zeros([S, N, K, D]) )
sample_gene_gauss_ass = tf.assign( sample_gene_gauss, generative_gauss.sample(sample_shape=[S]) )
sample_p_z = tf.Variable( tf.zeros([S, N, K]) )
sample_p_z_ass = tf.assign( sample_p_z, tf.to_float( p_z.sample(sample_shape=[S]) ) )
sample_p_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_p_mu_ass = tf.assign( sample_p_mu, p_mu.sample(sample_shape=[S]) )
sample_p_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_p_Lambda_ass = tf.assign( sample_p_Lambda, p_Lambda.sample(sample_shape=[S]) )
sample_p_pi = tf.Variable( tf.ones([S, K]) )
sample_p_pi_ass = tf.assign( sample_p_pi, p_pi.sample(sample_shape=[S]) )
sample_q_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_q_mu_ass = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )
sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )
sample_q_z = tf.Variable( tf.zeros([S, N, K]) )
sample_q_z_ass = tf.assign( sample_q_z, tf.to_float( q_z.sample(sample_shape=[S]) ) )
sample_q_pi = tf. Variable( tf.ones([S, K]) )
sample_q_pi_ass = tf.assign( sample_q_pi, q_pi.sample(sample_shape=[S]) )


# logarithmic distributions
print('* logarithmic distributions *')
with tf.name_scope('LogDistributions'):
    log_gene_gauss = generative_gauss.log_prob(sample_gene_gauss_ass, name='log_gene_gauss')
    logpx = tf.reduce_sum( tf.multiply( tf.to_float( sample_p_z_ass ), log_gene_gauss ), axis=2, name='logpx' )		
    log_p_x = tf.reduce_sum( logpx, axis=1, name='log_p_x' )
    log_p_pi = p_pi.log_prob( sample_p_pi_ass, name='log_p_pi' )
    log_p_z = tf.reduce_sum( p_z.log_prob( sample_p_z_ass ), axis=1, name='log_p_z' )
    log_p_mu = tf.reduce_sum( p_mu.log_prob( sample_p_mu_ass ), axis=1, name='log_p_mu' )
    log_p_Lambda = tf.reduce_sum( p_Lambda.log_prob( sample_p_Lambda_ass ), axis=1, name='log_p_Lambda' )
    log_dirichlet = q_pi.log_prob( sample_q_pi_ass, name='log_dirichlet' )
    log_categorical = q_z.log_prob( sample_q_z_ass, name='log_categorical' )
    log_gauss = q_mu.log_prob( sample_q_mu_ass, name='log_gauss' )
    log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
    log_q_pi = log_dirichlet 
    log_q_z = tf.reduce_sum( log_categorical, axis=1, name='log_q_z' )
    log_q_mu = tf.reduce_sum( log_gauss, axis=1, name='log_q_mu' )
    log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
    log_p = tf.add( tf.add( tf.add( tf.add( log_p_x, log_p_z ), log_p_pi ), log_p_mu ), log_p_Lambda, name='log_p' ) 
    log_q = tf.add( tf.add( tf.add( log_q_z, log_q_mu ), log_q_pi ), log_q_Lambda, name='log_q' ) 
    log_loss = tf.subtract( log_p, log_q, name='log_loss' )


# Gradient
print('* Gradient *')
with tf.name_scope('Gradient'):
    grad_q_z = []
    grad_q_pi = []
    grad_q_mu = []
    grad_q_muLambda = []
    grad_q_Lambda = []
    grad_q_nu = []
    for i in range(S):
        grad_q_z.append( tf.gradients(log_q[i], lambda_z) )
        grad_q_pi.append( tf.gradients(log_q[i], lambda_pi) )
        grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
        grad_q_muLambda.append( tf.gradients(log_q[i], lambda_muLambda) )
        grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda) )
        grad_q_nu.append( tf.gradients(log_q[i], lambda_nu) )
        
    grad_q_z_tf = tf.convert_to_tensor(grad_q_z)
    grad_q_pi_tf = tf.convert_to_tensor(grad_q_pi)
    grad_q_mu_tf = tf.convert_to_tensor(grad_q_mu)
    grad_q_muLambda_tf = tf.convert_to_tensor(grad_q_muLambda)
    grad_q_Lambda_tf = tf.convert_to_tensor(grad_q_Lambda)
    grad_q_nu_tf = tf.convert_to_tensor(grad_q_nu)


print("* Sample mean *")
# Sample mean(Montecarlo Approximation)
element_wise_product_z = []
element_wise_product_pi = []
element_wise_product_mu = []
element_wise_product_muLambda = []
element_wise_product_Lambda = []
element_wise_product_nu = []
for j in range(S):
    element_wise_product_z.append( tf.multiply(grad_q_z[j], log_loss[j]) )
    element_wise_product_pi.append( tf.multiply(grad_q_pi[j], log_loss[j]) )
    element_wise_product_mu.append( tf.multiply(grad_q_mu[j], log_loss[j]) )
    element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda[j], log_loss[j]) )
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda[j], log_loss[j]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu[j], log_loss[j]) )
sample_mean_z = tf.reduce_mean( element_wise_product_z, axis = 0 )[0]    
sample_mean_pi = tf.reduce_mean( element_wise_product_pi, axis = 0 )[0]
sample_mean_mu = tf.reduce_mean( element_wise_product_mu, axis = 0 )[0]
sample_mean_muLambda = tf.reduce_mean( element_wise_product_muLambda, axis = 0 )[0]
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda, axis = 0 )[0]
sample_mean_nu = tf.reduce_mean( element_wise_product_nu, axis = 0 )[0]


print("* Update parameters *")
# Update variational parameters
lambda_z_update = tf.assign(lambda_z, tf.add(lambda_z, tf.multiply(rho, sample_mean_z)) )
lambda_pi_update = tf.assign(lambda_pi, tf.add(lambda_pi, tf.multiply(rho, sample_mean_pi)) )
lambda_mu_update = tf.assign(lambda_mu, tf.add(lambda_mu, tf.multiply(rho, sample_mean_mu)) )
lambda_muLambda_update = tf.assign(lambda_muLambda, tf.add(lambda_muLambda, tf.multiply(rho, sample_mean_muLambda)) )
lambda_Lambda_update = tf.assign(lambda_Lambda, tf.add(lambda_Lambda, tf.multiply(rho, sample_mean_Lambda)) )
lambda_nu_update = tf.assign(lambda_nu, tf.add(lambda_nu, tf.multiply(rho, sample_mean_nu)) )


print("* Care values *")
# Care Values
_lambda_pi = []
_lambda_pi.append( tf.split(lambda_pi, K, 0) )
k=0
while(k < K):
    _lambda_pi[0][k] = tf.cond( tf.less_equal( _lambda_pi[0][k][0], 0.0 ), lambda: tf.abs( tf.multiply(0.5, _lambda_pi[0][k]) ), lambda: _lambda_pi[0][k] )
    k = k + 1
if(k == K):
    lambda_pi_update = tf.assign( lambda_pi, tf.concat(_lambda_pi[0], 0) )
del _lambda_pi[:]
gc.collect()

_lambda_z = []
_lambda_z.append( tf.split(lambda_z, N, 0) )
n=0
#print(n)
while(n < N):
    k=0
    while(k < K):
        #tf.less_equal( lambda_z[0][n][0][k], 0.0 )
        _lambda_z[0][n] = tf.cond( tf.less_equal( _lambda_z[0][n][0][k], 0.0 ), lambda: PGMethod(_lambda_z[0][n], [1, K]), lambda: _lambda_z[0][n] )
        k = k + 1
    _lambda_z[0][n] = tf.cond( tf.logical_or( tf.less_equal( tf.reduce_sum( _lambda_z[0][n] ), 0.9999999 ), tf.greater_equal( tf.reduce_sum( _lambda_z[0][n] ), 1.0000001 ) ), lambda: PGMethod( _lambda_z[0][n], [1, K] ), lambda: _lambda_z[0][n] )
    n = n + 1
if(n == N):
    lambda_z_update = tf.assign( lambda_z, tf.concat(_lambda_z[0], 0) )
del _lambda_z[:]
gc.collect()

#_lambda_muLambda = []
#_lambda_muLambda.append( tf.split(lambda_muLambda_update, K, 0) )		# split lambda_muLambda with each classes
##print(_lambda_muLambda[0][1][0])
#eig_lambda_muLambda = []
#blocks_diag_lambda_muLambda = []
#for k in range(K):
#    eig_lambda_muLambda.append( tf.self_adjoint_eigvals(_lambda_muLambda[0][k][0]) )		# eigen values for lambda_muLambda_{k}
#    for d in range(D):
#        eig_lambda_muLambda[k] = tf.cond( tf.less(eig_lambda_muLambda[k][d], 0.0), lambda: tf.abs(eig_lambda_muLambda[k]), lambda: eig_lambda_muLambda[k] ) 
#    blocks_diag_lambda_muLambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_muLambda[k]).to_dense() )
##lambda_muLambda_update = tf.reshape( tf.concat( [tf.concat( [blocks_diag_lambda_muLambda[0], blocks_diag_lambda_muLambda[1]], 0), blocks_diag_lambda_muLambda[2]], 0 ), [K, D, D] )
#lambda_muLambda_update = tf.reshape( blocks_diag_lambda_muLambda, [K, D, D] )
for k in range(K):
    lambda_muLambda_update = tf.assign( lambda_muLambda, tf.cond( tf.less(lambda_muLambda_update[k], 0.0), lambda: tf.abs(lambda_muLambda_update), lambda: lambda_muLambda_update ) )

_lambda_Lambda = []
_lambda_Lambda.append( tf.split(lambda_Lambda_update, K, 0) )			# split lambda_Lambda with each classes
eig_lambda_Lambda = []
eig_vec_lambda_Lambda = []
blocks_diag_lambda_Lambda = []
_lambda_Lambda_update = []
dets = []
off_diagonal = tf.ones([K, D, D]) - unit_matrix
epsilon = []
blocks_off_diagonal = []
for k in range(K):
    eig_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[0] )
    eig_vec_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[1] )
    for d in range(D):
        eig_lambda_Lambda[k] = tf.cond( tf.less(eig_lambda_Lambda[k][d], 0.0), lambda: tf.abs(eig_lambda_Lambda[k]), lambda: eig_lambda_Lambda[k] )
    epsilon.append( tf.multiply( 0.1, tf.exp( -tf.reduce_prod(eig_lambda_Lambda[k]) ) ) )
    blocks_off_diagonal.append( tf.multiply( epsilon[k], off_diagonal[k] ) )
    blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() )
    _lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] + blocks_off_diagonal[k] )
    #_lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] )
    #_lambda_Lambda_update.append( tf.matmul( blocks_diag_lambda_Lambda[k], tf.matmul( eig_vec_lambda_Lambda[k], tf.matrix_inverse(eig_vec_lambda_Lambda[k]) ) ) )
    #blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() + _lambda_Lambda[0][k][0] - tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part(_lambda_Lambda[0][k][0]) ).to_dense() )
    dets.append( tf.matrix_determinant(_lambda_Lambda_update[k]) )
    #lambda_Lambda_update[k] = tf.assign( lambda_Lambda, tf.cond( tf.less_equal( tf.matrix_determinant(lambda_Lambda_update[k]), 0.0 ), lambda: tf.reshape( _lambda_Lambda_update[k] , [D, D] ), lambda: _lambda_Lambda[k] ) )
    #cond = tf.less_equal( tf.matrix_determinant(lambda_Lambda_update[k]), 0.0 )
lambda_Lambda_update = tf.assign( lambda_Lambda, tf.reshape( _lambda_Lambda_update , [K, D, D] ) )

_lambda_nu = []
for k in range(K):
    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], tf.to_float(D - 1)), lambda: tf.to_float(D), lambda: lambda_nu_update[k] ) )
lambda_nu_update = tf.assign( lambda_nu, tf.convert_to_tensor(_lambda_nu) )


# Session
sess = tf.Session()
#summary_writer = tf.summary.FileWriter('graph_BBVI', tf.get_default_graph())
#if not os.path.exists('graph_BBVI'):
#    os.makedirs('graph_BBVI')
#time.sleep(5)
#summary_op = tf.summary.merge_all()


# Initiaize seesion
init = tf.global_variables_initializer()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init)
#print( sess.run(lambda_Lambda) )
#print( sess.run( p_Lambda.sample(sample_shape=[1]) ) )
#print( sess.run( tf.matrix_inverse( p_Lambda.sample(sample_shape=[1]) ) ) )
#print( sess.run( q_mu.sample(sample_shape=[1]) ) )
#print( sess.run( q_Lambda.sample(sample_shape=[1]) ) )
#print( sess.run( mu_gene ) )
#print( sess.run( sample_gene_gauss_ass ) )
#print( sess.run( sample_p_mu_ass ) )
#print( sess.run( sample_p_z_ass ) )
#print( sess.run( sample_p_pi_ass ) )
#print( sess.run( sample_q_mu_ass ) )
#print( sess.run( sample_q_z_ass ) )
#print( sess.run( sample_q_pi_ass ) )
#print( sess.run( tf.multiply( tf.to_float( sample_p_z_ass ), log_gene_gauss ) ) )
#print( sess.run( logpx ) )
#print( sess.run( log_p_x ) )
#print( sess.run( p_mu.log_prob( sample_p_mu ) ) )
#print( sess.run( log_p_mu ) )
#print( sess.run( log_p_Lambda ) )
#print( sess.run( log_p_pi ) )
#print( sess.run( log_p_z ) )
#print( sess.run( log_dirichlet ) )
#print( sess.run( log_categorical ) )
#print( sess.run( log_gauss ) )
#print( sess.run( log_wishart ) )
#print( sess.run( log_q_pi ) )
#print( sess.run( tf.reduce_sum( log_gauss, axis=1 ) ) )
#print( sess.run( log_q_mu ) )
#print( sess.run( log_q_Lambda ) )
#print( sess.run( log_q_z ) )
#print( sess.run( log_p ) )
#print( sess.run( log_q ) )
#print( sess.run( log_loss ) )
#print( sess.run( grad_q_z ) )
#print( sess.run( grad_q_pi ) )
#print( sess.run( grad_q_mu ) )
#print( sess.run( grad_q_muLambda ) )
#print( sess.run( grad_q_Lambda ) )
#print( sess.run( grad_q_nu ) )
#print( sess.run( sample_mean_z ) )
#print( sess.run( sample_mean_pi ) )
#print( sess.run( sample_mean_mu ) )
#print( sess.run( sample_mean_muLambda ) )
#print( sess.run( sample_mean_Lambda ) )
#print( sess.run( sample_mean_nu ) )
#print( sess.run( [lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update] ) )
#print( sess.run( lambda_Lambda_update ) )
#print( sess.run( lambda_nu_update ) )
#print( sess.run( _lambda_muLambda[0][1][0] ) )
#print( sess.run( tf.self_adjoint_eigvals(_lambda_muLambda[0][1][0]) ) )
#print( sess.run( eig_lambda_muLambda ) )
#print( sess.run( blocks_diag_lambda_muLambda ) )
#print( sess.run( tf.reshape( tf.concat( [tf.concat( [blocks_diag_lambda_muLambda[0], blocks_diag_lambda_muLambda[1]], 0), blocks_diag_lambda_muLambda[2]], 0 ), [K, D, D] ) ) )
#print( sess.run( lambda_muLambda_update ) )
#print( sess.run( tf.self_adjoint_eig( _lambda_Lambda[0][1][0] ) ) )
#print( sess.run( eig_lambda_Lambda ) )
#print( sess.run( eig_vec_lambda_Lambda[1] ) )
#print( sess.run( blocks_diag_lambda_Lambda ) )
#print( sess.run( _lambda_Lambda_update ) )
#print( sess.run( [blocks_diag_lambda_Lambda, _lambda_Lambda_update, dets] ) )
#print( sess.run( blocks_diag_lambda_Lambda ) )
#print( sess.run( blocks_off_diagonal ) )
#print( sess.run( _lambda_Lambda_update ) )
#print( sess.run( [lambda_Lambda_update, dets] ) )
#print( sess.run( lambda_nu_update ) )


for epoch in range(10):
    print( sess.run( [lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update, dets] ) )
    #print( sess.run( [lambda_muLambda_update, lambda_Lambda_update] ) )
    #print( sess.run( [lambda_Lambda_update, lambda_nu_update, dets] ) )
    #print( sess.run( [lambda_Lambda_update, dets, epsilon] ) )





