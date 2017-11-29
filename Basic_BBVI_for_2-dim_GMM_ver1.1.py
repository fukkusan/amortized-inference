import tensorflow as tf
import numpy as np
import Input_data_2dGMM as inp
import gc




# Projected Gradient Method
def PGMethod(vector_subspace, element_numbers):
    normal_vector = tf.ones(element_numbers)
    coefficient = tf.reduce_sum( tf.multiply(normal_vector, vector_subspace) )
    norm = tf.norm(normal_vector)
    oriented_vector = tf.multiply( coefficient, tf.divide(normal_vector, norm) )
    element_sum = tf.reduce_sum( tf.abs(oriented_vector) )
    vector_constrainted = tf.divide( oriented_vector, element_sum )
    
    
    return vector_constrainted




# * Initialize *
print("* Initialize *")
# Constants
N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample
_alpha = 0.0
_beta = tf.sqrt(0.1)
_gamma = 1.0
num_epochs = 10
num_samples = 100

# Observable
input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]

# Input placeholder
X = tf.placeholder(tf.float32, shape = [N, D])


# Parameters
# Hyper parameters
unit_matrices = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
off_diagonal = tf.subtract( tf.ones([K, D, D]), unit_matrices )
hyper_alpha_mean = tf.constant(x_mean, shape=[K, D], dtype=tf.float32)
hyper_coe_alpha_var = tf.multiply( _beta, unit_matrices )
hyper_gamma = tf.constant(_gamma, shape=[K], dtype=tf.float32)
hyper_V = unit_matrices
hyper_nu = tf.constant(2.0, shape=[K])

# definition Variational parameters
lambda_mu = tf.Variable( tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_muLambda = tf.Variable( unit_matrices, dtype=tf.float32, trainable=True, name='lambda_muLambda' )
lambda_Lambda = tf.Variable( unit_matrices , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )
lambda_pi = tf.Variable( tf.ones([K])*K, dtype=tf.float32, trainable=True, name='lambda_pi' )
lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32, trainable=True, name='lambda_z' )

# Save previous variational parameters
prev_lambda_pi = tf.Variable( tf.ones(K)*K, dtype=tf.float32 )
prev_lambda_mu = tf.Variable( tf.ones([D, K]), dtype=tf.float32, trainable=True )
prev_lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32 )
prev_lambda_Lambda = tf.Variable( unit_matrices, dtype=tf.float32, trainable=True )
prev_lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True )

# Update counter and Learning parameter
update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )
rho = tf.Variable( 0.1, dtype=tf.float32, trainable=True )


# Distributions
# Approximated distributions
precision = tf.multiply( lambda_muLambda, unit_matrices )			# precision is constraited to be diagonal matrix
covariance_q_mu = tf.matrix_inverse( precision )
q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=lambda_mu, covariance_matrix=covariance_q_mu )
q_Lambda = tf.contrib.distributions.WishartFull( df=lambda_nu, scale=lambda_Lambda )
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z)			
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi)				

# Generative model
p_Lambda = tf.contrib.distributions.WishartFull( df=hyper_nu, scale=hyper_V )
_covariance_p_mu = tf.multiply( hyper_coe_alpha_var, p_Lambda.sample( sample_shape=[1] )[0] )
covariance_p_mu = tf.matrix_inverse( _covariance_p_mu )
p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=hyper_alpha_mean, covariance_matrix=covariance_p_mu )
p_pi = tf.contrib.distributions.Dirichlet(hyper_gamma)					
pi_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_pi.sample(sample_shape=[N]), lambda: p_pi.sample(sample_shape=[N]), name='pi_gene' )		# pi for Cat(z|pi) is sampled by p_pi firstly and q_pi secondly
mu_gene = tf.cond( tf.greater(update_counter[0], 0.0), lambda: q_mu.sample(sample_shape=[N]), lambda: p_mu.sample(sample_shape=[N]), name='mu_gene' )		# mu for Normal(x|mu,Lambda) is sampled by p_mu firstly and q_mu secondly
covariance_generative_gauss = tf.cond( tf.greater(update_counter[0], 0.0), lambda: tf.matrix_inverse( q_Lambda.sample( sample_shape=[N] ) ), lambda: tf.matrix_inverse( p_Lambda.sample( sample_shape=[N] ) ) )
p_z = tf.contrib.distributions.OneHotCategorical( pi_gene )
generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=mu_gene, covariance_matrix=covariance_generative_gauss )


# * Construct calculation graph for updating variational parameters *
# Sampling from approximated distributions
print("* Sampling *")
sample_gene_gauss = tf.Variable( tf.zeros([S, N, K, D]) )
x_gene = tf.reshape( [ X[:], X[:], X[:] ] , [N, K, D] )
_x_gene_sample = []
for i in range(S):
    _x_gene_sample.append( x_gene )
x_gene_sample = tf.reshape( _x_gene_sample, [S, N, K, D] )
sample_gene_gauss_ass = tf.assign( sample_gene_gauss, x_gene_sample )
sample_q_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_q_mu_ass = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )
sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )
sample_q_z = tf.Variable( tf.zeros([S, N, K]) )
sample_q_z_ass = tf.assign( sample_q_z, tf.to_float( q_z.sample(sample_shape=[S]) ) )
sample_q_pi = tf. Variable( tf.ones([S, K]) )
sample_q_pi_ass = tf.assign( sample_q_pi, q_pi.sample(sample_shape=[S]) )
del _x_gene_sample[:]
gc.collect()


# logarithmic distributions
print("* logarithmic distributions *")
with tf.name_scope('LogDistributions'):
    log_gene_gauss = generative_gauss.log_prob(sample_gene_gauss_ass, name='log_gene_gauss')
    logpx = tf.reduce_sum( tf.multiply( tf.to_float( sample_q_z_ass ), log_gene_gauss ), axis=2, name='logpx' )		
    log_p_x = tf.reduce_sum( logpx, axis=1, name='log_p_x' )
    log_p_pi = p_pi.log_prob( sample_q_pi_ass, name='log_p_pi' )
    log_p_z = tf.reduce_sum( p_z.log_prob( sample_q_z_ass ), axis=1, name='log_p_z' )
    log_p_mu = tf.reduce_sum( p_mu.log_prob( sample_q_mu_ass ), axis=1, name='log_p_mu' )
    log_p_Lambda = tf.reduce_sum( p_Lambda.log_prob( sample_q_Lambda_ass ), axis=1, name='log_p_Lambda' )
    
    log_gauss = q_mu.log_prob( sample_q_mu_ass, name='log_gauss' )
    log_q_mu = tf.reduce_sum( log_gauss, axis=1, name='log_q_mu' )
    log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
    log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
    log_dirichlet = q_pi.log_prob( sample_q_pi_ass, name='log_dirichlet' )
    log_q_pi = log_dirichlet
    log_categorical = q_z.log_prob( sample_q_z_ass, name='log_categorical' )
    log_q_z = tf.reduce_sum( log_categorical, axis=1, name='log_q_z' )
    log_p = tf.add( tf.add( tf.add( tf.add( log_p_x, log_p_z ), log_p_pi ), log_p_mu ), log_p_Lambda, name='log_p' ) 
    log_q = tf.add( tf.add( tf.add( log_q_z, log_q_mu ), log_q_pi ), log_q_Lambda, name='log_q' ) 
    log_loss = tf.subtract( log_p, log_q, name='log_loss' )


# Gradients
print("* Gradients *")
grad_q_mu = []
grad_q_muLambda = []
grad_q_Lambda = []
grad_q_nu = []
grad_q_z = []
grad_q_pi = []

for j in range(S):
    grad_q_mu.append( tf.gradients(log_q[j], lambda_mu)[0] )
    grad_q_muLambda.append( tf.gradients(log_q[j], lambda_muLambda)[0] )
    grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda)[0] )
    grad_q_nu.append( tf.gradients(log_q[j], lambda_nu)[0] )
    grad_q_z.append( tf.gradients(log_q[j], lambda_z)[0] )
    grad_q_pi.append( tf.gradients(log_q[j], lambda_pi)[0] )



# Sample mean
print("* Sample mean *")
element_wise_product_mu = []
element_wise_product_muLambda = []
element_wise_product_Lambda = []
element_wise_product_nu = []
element_wise_product_z = []
element_wise_product_pi = []
#print(log_loss[1])
#print(grad_q_mu[1])
for s in range(S):
    element_wise_product_mu.append( tf.multiply(grad_q_mu[s], log_loss[s]) )				# Why can use tf.multiply between different type tensors?
    element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda[s], log_loss[s]) )
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda[s], log_loss[s]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu[s], log_loss[s]) )
    element_wise_product_z.append( tf.multiply(grad_q_z[s], log_loss[s]) )
    element_wise_product_pi.append( tf.multiply(grad_q_pi[s], log_loss[s]) )
sample_mean_mu = tf.reduce_mean( element_wise_product_mu, axis = 0 )					
sample_mean_muLambda = tf.reduce_mean( element_wise_product_muLambda, axis = 0 )
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda, axis = 0 )
sample_mean_nu = tf.reduce_mean( element_wise_product_nu, axis = 0 )
sample_mean_z = tf.reduce_mean( element_wise_product_z, axis = 0 )    
sample_mean_pi = tf.reduce_mean( element_wise_product_pi, axis = 0 )


# Update variational parameters
print("* Update variational parameters *")
delta_mu = tf.multiply(rho, sample_mean_mu)
lambda_mu_update = tf.assign_add( lambda_mu, delta_mu )
delta_muLambda = tf.multiply(rho, sample_mean_muLambda)
lambda_muLambda_update = tf.assign_add( lambda_muLambda, delta_muLambda )
delta_Lambda = tf.multiply(rho, sample_mean_Lambda)
lambda_Lambda_update = tf.assign_add( lambda_Lambda, delta_Lambda )
delta_nu = tf.multiply(rho, sample_mean_nu)
lambda_nu_update = tf.assign_add( lambda_nu, delta_nu )
delta_z = tf.multiply(rho, sample_mean_z)
lambda_z_update = tf.assign_add( lambda_z, delta_z )
delta_pi = tf.multiply(rho, sample_mean_pi)
lambda_pi_update = tf.assign_add( lambda_pi, delta_pi )


# Update time_step and learning parameter
update_counter = tf.assign( update_counter, tf.add(update_counter, tf.ones(1)) )
rho = tf.minimum( 0.1, tf.divide(1, tf.to_float(update_counter)) )



# Caring variational parameters
print("* Care lambda_muLambda *")
# * Diagonalize lambda_muLambda and transform negative eigen values to positve ones *
eig_val_lambda_muLambda = tf.self_adjoint_eigvals(lambda_muLambda_update)		# eigenvalues of lambda_muLambda
eig_val_lambda_muLambda_modify = tf.abs( eig_val_lambda_muLambda )
diagonalized_muLambda = tf.contrib.linalg.LinearOperatorDiag( eig_val_lambda_muLambda_modify ).to_dense()
noize_diag_matrix = tf.multiply(tf.sqrt(0.1), unit_matrices)
pred_lambda_muLambda = tf.cast( tf.greater( eig_val_lambda_muLambda, tf.zeros([K, D]) ), dtype=tf.float32 )			# The condition of tf.cond must compare scalars but this tf.less_equal gives boolean tensor with shape [K, D].
_lambda_muLambda_cared = tf.cond( tf.equal( tf.reduce_sum( pred_lambda_muLambda ), tf.multiply( tf.to_float(K), tf.to_float(D) ) ), lambda: lambda_muLambda_update, lambda: tf.add( diagonalized_muLambda, noize_diag_matrix ) )		# Sum of pred_lambda_muLambda is K*D if all eigenvalues are positive.
lambda_muLambda_cared = tf.assign( lambda_muLambda, _lambda_muLambda_cared )
zero_eig_muLambda = tf.assert_none_equal( eig_val_lambda_muLambda_modify, tf.zeros([K, D]) )


print("* Care lambda_Lambda *")
# * Diagonalize lambda_Lambda and transform negative values to positive ones *
eig_val_lambda_Lambda = tf.self_adjoint_eigvals(lambda_Lambda_update)		# eigenvalues of lambda_Lambda
eig_val_lambda_Lambda_modify = tf.abs( eig_val_lambda_Lambda )
diagonalized_Lambda = tf.contrib.linalg.LinearOperatorDiag( eig_val_lambda_Lambda_modify ).to_dense()
pred_lambda_Lambda = tf.cast( tf.greater( eig_val_lambda_Lambda, tf.zeros([K, D]) ), dtype=tf.float32 )			# The condition of tf.cond must compare scalars but this tf.less_equal gives boolean tensor with shape [K, D].
_lambda_Lambda_cared = tf.cond( tf.equal( tf.reduce_sum( pred_lambda_Lambda ), tf.multiply( tf.to_float(K), tf.to_float(D) ) ), lambda: lambda_Lambda_update, lambda: tf.add( diagonalized_Lambda, noize_diag_matrix ) )		# Sum of pred_lambda_Lambda is K*D if all eigenvalues are positive.
lambda_Lambda_cared = tf.assign( lambda_Lambda, _lambda_Lambda_cared )
zero_eig_Lambda = tf.assert_none_equal( eig_val_lambda_Lambda_modify, tf.zeros([K, D]) )


print("* Care lambda_nu *")
# * lambda_nu must be lager than D-1 *
dimension = tf.constant(D, shape=[K], dtype=tf.float32)
pred_lambda_nu = tf.cast( tf.greater( lambda_nu_update, tf.subtract( dimension, tf.ones([K]) ) ), dtype=tf.float32 )
_lambda_nu_cared = tf.cond( tf.equal( tf.reduce_sum( pred_lambda_nu ), tf.to_float(K) ), lambda: lambda_nu_update, lambda: tf.add( tf.abs( lambda_nu_update ), tf.subtract( dimension, tf.ones([K]) ) ) )
lambda_nu_cared = tf.assign( lambda_nu, _lambda_nu_cared )


print("* Care lambda_z *")
# * lambda_z >= 0 and sum(lambda_z_{nk}, k) = 1
_lambda_z = []
_lambda_z.append( tf.split(lambda_z_update, N, 0) )
for n in range(N):
    _lambda_z[0][n] = PGMethod(_lambda_z[0][n], [1, K])
_lambda_z_update = tf.concat( _lambda_z[0], 0 )
pred_lambda_z0 = tf.cast( tf.greater( lambda_z_update, tf.zeros([N, K]) ), dtype=tf.float32 )		# lambda_z > 0
pred_lambda_z1 = tf.cast( tf.logical_and( tf.greater( tf.reduce_sum( lambda_z_update, axis=1 ), tf.constant(0.999999, shape=[N]) ), tf.less( tf.reduce_sum( lambda_z_update, axis=1 ), tf.constant(1.00001, shape=[N]) ) ), dtype=tf.float32 )		# sum(lambda_z, k)=1
_lambda_z_cared = tf.cond( tf.logical_and( tf.equal( tf.reduce_sum( pred_lambda_z0 ), tf.multiply( tf.to_float(N), tf.to_float(K) ) ), tf.equal( tf.reduce_sum( pred_lambda_z1 ), tf.to_float(N) ) ), lambda: lambda_z_update, lambda: _lambda_z_update )
lambda_z_cared = tf.assign( lambda_z, _lambda_z_cared )
del _lambda_z[:]
gc.collect()


print("* Care lambda_pi *")
# * lambda_pi >= 0 *
pred_lambda_pi = tf.cast( tf.greater( lambda_pi_update, tf.zeros([K]) ), dtype=tf.float32 )
_lambda_pi_cared = tf.cond( tf.equal( tf.reduce_sum(pred_lambda_pi), tf.to_float(K) ), lambda: lambda_pi_update, lambda: tf.abs(tf.multiply( 0.5, lambda_pi_update )) )
lambda_pi_cared = tf.assign( lambda_pi, _lambda_pi_cared )



sess = tf.Session()
init = tf.global_variables_initializer()
cnt = 0
sess.run(init)

for epoch in range(10):
    result = sess.run( [zero_eig_Lambda, zero_eig_muLambda, lambda_mu_update, lambda_muLambda_cared, lambda_Lambda_cared, lambda_nu_cared, lambda_z_cared, lambda_pi_cared], feed_dict={X:x} )
    cnt = cnt + 1
    print(cnt)
    print(result)





