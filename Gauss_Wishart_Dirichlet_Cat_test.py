import tensorflow as tf
import gc
import csv
from tensorflow.python import debug
#import ProjectedGradientMethod as pgm



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
# Constant
N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample


# definition Variational parameters
unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
off_diagonal = tf.ones([K, D, D]) - unit_matrix 
lambda_mu = tf.Variable( tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_muLambda = tf.Variable( tf.ones([K]), dtype=tf.float32, trainable=True, name='lambda_muLambda' )
lambda_Lambda = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )
lambda_pi = tf.Variable( tf.ones([K])*K, dtype=tf.float32, trainable=True, name='lambda_pi' )
lambda_z = tf.Variable( tf.ones([N, K])/K, dtype=tf.float32, trainable=True, name='lambda_z' )


# Update counter and Learning parameter
update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )
#rho = tf.Variable( tf.constant(0.1, shape=[1]), dtype=tf.float32, trainable=True )
rho = tf.Variable( 0.1, dtype=tf.float32, trainable=True )


# Approximated distributions
precision = []
for k in range(K):
    precision.append( tf.multiply( lambda_muLambda[k], unit_matrix[k] ) )
precision_tensor = tf.convert_to_tensor(precision)
covariance_q_mu = tf.matrix_inverse( precision_tensor )
del precision[:]
gc.collect()
q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=lambda_mu, covariance_matrix=covariance_q_mu )
q_Lambda = tf.contrib.distributions.WishartFull( df=lambda_nu, scale=lambda_Lambda )
q_z = tf.contrib.distributions.OneHotCategorical(lambda_z)			
q_pi = tf.contrib.distributions.Dirichlet(lambda_pi)				



# * Construct calculation graph for updating variational parameters *
# Sampling from approximated distributions
print("* Sampling *")
sample_q_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_q_mu_ass = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )
sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )
sample_q_z = tf.Variable( tf.zeros([S, N, K]) )
sample_q_z_ass = tf.assign( sample_q_z, tf.to_float( q_z.sample(sample_shape=[S]) ) )
sample_q_pi = tf. Variable( tf.ones([S, K]) )
sample_q_pi_ass = tf.assign( sample_q_pi, q_pi.sample(sample_shape=[S]) )


# logarithmic distributions
print("* logarithmic distributions *")
log_gauss = q_mu.log_prob( sample_q_mu_ass, name='log_gauss' )
log_q_mu = tf.reduce_sum( log_gauss, axis=1, name='log_q_mu' )
log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
log_dirichlet = q_pi.log_prob( sample_q_pi_ass, name='log_dirichlet' )
log_q_pi = log_dirichlet
log_categorical = q_z.log_prob( sample_q_z_ass, name='log_categorical' )
log_q_z = tf.reduce_sum( log_categorical, axis=1, name='log_q_z' )
log_q = tf.add( tf.add( tf.add( log_q_mu, log_q_Lambda ), log_q_z ), log_q_pi )										# Is any problem?


# Gradients
print("* Gradients *")
grad_q_mu = []
grad_q_muLambda = []
grad_q_Lambda = []
grad_q_nu = []
grad_q_z = []
grad_q_pi = []
for i in range(S):
    grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
    grad_q_muLambda.append( tf.gradients(log_q[i], lambda_muLambda) )
    grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda) )
    grad_q_nu.append( tf.gradients(log_q[i], lambda_nu) )
    grad_q_z.append( tf.gradients(log_q[i], lambda_z) )
    grad_q_pi.append( tf.gradients(log_q[i], lambda_pi) )
grad_q_mu_tf = tf.convert_to_tensor(grad_q_mu)				#
grad_q_muLambda_tf = tf.convert_to_tensor(grad_q_muLambda)	#
grad_q_Lambda_tf = tf.convert_to_tensor(grad_q_Lambda)		#
grad_q_nu_tf = tf.convert_to_tensor(grad_q_nu)				#
grad_q_z_tf = tf.convert_to_tensor(grad_q_z)
grad_q_pi_tf = tf.convert_to_tensor(grad_q_pi)
del grad_q_Lambda[:]
del grad_q_nu[:]
del grad_q_muLambda[:]
del grad_q_mu[:]
del grad_q_z[:]
del grad_q_pi[:]
gc.collect()



# Sample mean
print("* Sample mean *")
element_wise_product_mu = []
element_wise_product_muLambda = []
element_wise_product_Lambda = []
element_wise_product_nu = []
element_wise_product_z = []
element_wise_product_pi = []
for j in range(S):
    element_wise_product_mu.append( tf.multiply(grad_q_mu_tf[j], log_q[j]) )				# grad_q_mu_tf[j][0]?
    element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda_tf[j], log_q[j]) )
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda_tf[j], log_q[j]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu_tf[j], log_q[j]) )
    element_wise_product_z.append( tf.multiply(grad_q_z_tf[j], log_q[j]) )
    element_wise_product_pi.append( tf.multiply(grad_q_pi_tf[j], log_q[j]) )
element_wise_product_mu_tf = tf.convert_to_tensor( element_wise_product_mu )				#
element_wise_product_muLambda_tf = tf.convert_to_tensor( element_wise_product_muLambda )	#
element_wise_product_Lambda_tf = tf.convert_to_tensor( element_wise_product_Lambda )		#
element_wise_product_nu_tf = tf.convert_to_tensor( element_wise_product_nu )				#
element_wise_product_z_tf = tf.convert_to_tensor( element_wise_product_z )
element_wise_product_pi_tf = tf.convert_to_tensor( element_wise_product_pi )
sample_mean_mu = tf.reduce_mean( element_wise_product_mu_tf, axis = 0 )[0]					
sample_mean_muLambda = tf.reduce_mean( element_wise_product_muLambda_tf, axis = 0 )[0]
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda_tf, axis = 0 )[0]
sample_mean_nu = tf.reduce_mean( element_wise_product_nu_tf, axis = 0 )[0]
sample_mean_z = tf.reduce_mean( element_wise_product_z_tf, axis = 0 )[0]    
sample_mean_pi = tf.reduce_mean( element_wise_product_pi_tf, axis = 0 )[0]
del element_wise_product_Lambda[:]
del element_wise_product_nu[:]
del element_wise_product_muLambda[:]
del element_wise_product_mu[:]
del element_wise_product_z[:]
del element_wise_product_pi[:]
gc.collect()


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


# Caring variational parameters
print("* Care lambda_muLambda *")
_lambda_muLambda = []
_lambda_muLambda.append( tf.split(lambda_muLambda_update, K, 0) )		# split lambda_muLambda with each classes
for k in range(K):
    _lambda_muLambda[0][k] = tf.cond( tf.less_equal(_lambda_muLambda[0][k][0], 0.0), lambda: tf.abs(_lambda_muLambda[0][k]), lambda: _lambda_muLambda[0][k] )
lambda_muLambda_update = tf.assign( lambda_muLambda, tf.concat(_lambda_muLambda[0], 0) )
#lambda_muLambda_cared = tf.assign( lambda_muLambda, tf.concat(_lambda_muLambda[0], 0) )
#lambda_muLambda_cared = tf.concat(_lambda_muLambda[0], 0)

print("* Care lambda_Lambda *")
_lambda_Lambda = []
_lambda_Lambda.append( tf.split(lambda_Lambda_update, K, 0) )			# split lambda_Lambda with each classes
eig_lambda_Lambda = []
eig_vec_lambda_Lambda = []
prev_eig_lambda_Lambda = []
blocks_diag_lambda_Lambda = []
_lambda_Lambda_update = []
for k in range(K):
    offDiag_lambda_Lambda = tf.multiply( tf.slice( _lambda_Lambda[0][k][0], [1, 0], [1, 1] ), off_diagonal[k] )
    _lambda_Lambda[0][k] = tf.add( tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( _lambda_Lambda[0][k] ) ).to_dense(), offDiag_lambda_Lambda )
    eig_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[0] )
    prev_eig_lambda_Lambda.append( eig_lambda_Lambda[k] )
    eig_vec_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[1] )
    for d in range(D):
        eig_lambda_Lambda[k] = tf.cond( tf.less_equal(eig_lambda_Lambda[k][d], 0.0), lambda: tf.abs(eig_lambda_Lambda[k]), lambda: eig_lambda_Lambda[k] )
    blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() )
    _lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] )
reshape_lambda_Lambda_update = tf.reshape( tf.convert_to_tensor(_lambda_Lambda_update) , [K, D, D] )
for k in range(K):
    for d in range(D):
        lambda_Lambda_update = tf.assign(lambda_Lambda, tf.cond( tf.less_equal( prev_eig_lambda_Lambda[k][d], tf.zeros(D)[d] ), lambda: reshape_lambda_Lambda_update , lambda: lambda_Lambda_update ) )
        #lambda_Lambda_cared = tf.assign(lambda_Lambda, tf.cond( tf.less_equal( prev_eig_lambda_Lambda[k][d], tf.zeros(D)[d] ), lambda: reshape_lambda_Lambda_update , lambda: lambda_Lambda_update ) )
        #lambda_Lambda_cared = tf.cond( tf.less_equal( prev_eig_lambda_Lambda[k][d], tf.zeros(D)[d] ), lambda: reshape_lambda_Lambda_update , lambda: lambda_Lambda_update )
#lambda_Lambda_update = tf.assign( lambda_Lambda, tf.reshape( tf.convert_to_tensor(_lambda_Lambda_update) , [K, D, D] ) )		# you should use tf.assign for updating parameter
#lambda_Lambda_update = tf.assign( lambda_Lambda, reshape_lambda_Lambda_update )
del _lambda_Lambda[:]
del eig_lambda_Lambda[:]
del eig_vec_lambda_Lambda[:]
del blocks_diag_lambda_Lambda[:]
del _lambda_Lambda_update[:]
del prev_eig_lambda_Lambda[:]
gc.collect()

print("* Care lambda_nu *")
_lambda_nu = []
dimension = tf.constant(D, dtype=tf.float32)
for k in range(K):
    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], dimension), lambda: dimension, lambda: lambda_nu_update[k] ) )
lambda_nu_update = tf.assign( lambda_nu, _lambda_nu )
#lambda_nu_cared = tf.assign( lambda_nu, _lambda_nu )
#lambda_nu_cared = _lambda_nu
del _lambda_nu[:]
gc.collect()

print("* Care lambda_z *")
_lambda_z = []
_lambda_z.append( tf.split(lambda_z_update, N, 0) )
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
    lambda_z_update = tf.assign( lambda_z, tf.concat(_lambda_z[0], 0) )
    #lambda_z_cared = tf.assign( lambda_z, tf.concat(_lambda_z[0], 0) )
    #lambda_z_cared = tf.concat(_lambda_z[0], 0)
del _lambda_z[:]
gc.collect()

print("* Care lambda_pi *")
_lambda_pi = []
_lambda_pi.append( tf.split(lambda_pi_update, K, 0) )
k=0
while(k < K):
    _lambda_pi[0][k] = tf.cond( tf.less_equal( _lambda_pi[0][k][0], 0.0 ), lambda: tf.abs( tf.multiply(0.5, _lambda_pi[0][k]) ), lambda: _lambda_pi[0][k] )
    k = k + 1
if(k == K):
    lambda_pi_update = tf.assign( lambda_pi, tf.concat(_lambda_pi[0], 0) )
    #lambda_pi_cared = tf.assign( lambda_pi, tf.concat(_lambda_pi[0], 0) )
    #lambda_pi_cared = tf.concat(_lambda_pi[0], 0)
del _lambda_pi[:]
gc.collect()



# Session
cnt = 0
sess = tf.Session()
init = tf.global_variables_initializer()
#file = open( 'output.csv', 'w' )
#writer = csv.writer(file, lineterminator='\n')
#csvlist = []
sess.run(init)
#sess = debug.LocalCLIDebugWrapperSession(sess)
for epoch in range(100000):
    print( sess.run([lambda_mu_update, lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update, lambda_z_update, lambda_pi_update]) )
    #print( sess.run([lambda_mu_update, lambda_muLambda_cared, lambda_Lambda_cared, lambda_nu_cared, lambda_z_cared, lambda_pi_cared]) )
    #csvlist.append( lambda_Lambda_update[1][0][1].eval(session=sess) )
    cnt = cnt + 1
    print(cnt)
#writer.writerow(csvlist)
#file.close





