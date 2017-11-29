import tensorflow as tf
import gc




# * Initialize *
print("* Initialize *")
# Constant
N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample


# definition Variational parameters
unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
lambda_mu = tf.Variable( tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_muLambda = tf.Variable( tf.ones([K]), dtype=tf.float32, trainable=True, name='lambda_muLambda' )
lambda_Lambda = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )


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



# * Construct calculation graph for updating variational parameters *
# Sampling from approximated distributions
print("* Sampling *")
sample_q_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_q_mu_ass = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )
sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )


# logarithmic distributions
print("* logarithmic distributions *")
log_gauss = q_mu.log_prob( sample_q_mu_ass, name='log_gauss' )
log_q_mu = tf.reduce_sum( log_gauss, axis=1, name='log_q_mu' )
log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
log_q = tf.add( log_q_mu, log_q_Lambda )										# Is any problem?


# Gradients
print("* Gradients *")
grad_q_mu = []
grad_q_muLambda = []
grad_q_Lambda = []
grad_q_nu = []
for i in range(S):
    grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
    grad_q_muLambda.append( tf.gradients(log_q[i], lambda_muLambda) )
    grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda) )
    grad_q_nu.append( tf.gradients(log_q[i], lambda_nu) )
    #grad_q_mu.append( tf.gradients(log_q_mu[i], lambda_mu) )
    #grad_q_muLambda.append( tf.gradients(log_q_mu[i], lambda_muLambda) )
    #grad_q_Lambda.append( tf.gradients(log_q_Lambda[i], lambda_Lambda) )
    #grad_q_nu.append( tf.gradients(log_q_Lambda[i], lambda_nu) )
grad_q_mu_tf = tf.convert_to_tensor(grad_q_mu)				#
grad_q_muLambda_tf = tf.convert_to_tensor(grad_q_muLambda)	#
grad_q_Lambda_tf = tf.convert_to_tensor(grad_q_Lambda)		#
grad_q_nu_tf = tf.convert_to_tensor(grad_q_nu)				#


# Sample mean
print("* Sample mean *")
element_wise_product_mu = []
element_wise_product_muLambda = []
element_wise_product_Lambda = []
element_wise_product_nu = []
for j in range(S):
    element_wise_product_mu.append( tf.multiply(grad_q_mu_tf[j], log_q[j]) )
    element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda_tf[j], log_q[j]) )
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda_tf[j], log_q[j]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu_tf[j], log_q[j]) )
    #element_wise_product_mu.append( tf.multiply(grad_q_mu_tf[j], log_q_mu[j]) )
    #element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda_tf[j], log_q_mu[j]) )
    #element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda_tf[j], log_q_Lambda[j]) )
    #element_wise_product_nu.append( tf.multiply(grad_q_nu_tf[j], log_q_Lambda[j]) )
element_wise_product_mu_tf = tf.convert_to_tensor( element_wise_product_mu )				#
element_wise_product_muLambda_tf = tf.convert_to_tensor( element_wise_product_muLambda )	#
element_wise_product_Lambda_tf = tf.convert_to_tensor( element_wise_product_Lambda )		#
element_wise_product_nu_tf = tf.convert_to_tensor( element_wise_product_nu )				#
sample_mean_mu = tf.reduce_mean( element_wise_product_mu_tf, axis = 0 )[0]
sample_mean_muLambda = tf.reduce_mean( element_wise_product_muLambda_tf, axis = 0 )[0]
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda_tf, axis = 0 )[0]
sample_mean_nu = tf.reduce_mean( element_wise_product_nu_tf, axis = 0 )[0]


# Update variational parameters
print("* Update variational parameters *")
delta_mu = tf.multiply(rho, sample_mean_mu)
lambda_mu_update = tf.assign( lambda_mu, tf.add(lambda_mu, delta_mu ) )
delta_muLambda = tf.multiply(rho, sample_mean_muLambda)
lambda_muLambda_update = tf.assign( lambda_muLambda, tf.add(lambda_muLambda, delta_muLambda) )
delta_Lambda = tf.multiply(rho, sample_mean_Lambda)
lambda_Lambda_update = tf.assign( lambda_Lambda, tf.add( lambda_Lambda, delta_Lambda ) )
delta_nu = tf.multiply(rho, sample_mean_nu)
lambda_nu_update = tf.assign( lambda_nu, tf.add( lambda_nu, delta_nu ) )


# Caring variational parameters
print("* Care lambda_muLambda *")
_lambda_muLambda = []
_lambda_muLambda.append( tf.split(lambda_muLambda_update, K, 0) )		# split lambda_muLambda with each classes
for k in range(K):
    _lambda_muLambda[0][k] = tf.cond( tf.less_equal(_lambda_muLambda[0][k][0], 0.0), lambda: tf.abs(_lambda_muLambda[0][k]), lambda: _lambda_muLambda[0][k] )
lambda_muLambda_update = tf.assign( lambda_muLambda, tf.concat(_lambda_muLambda[0], 0) )

print("* Care lambda_Lambda *")
_lambda_Lambda = []
_lambda_Lambda.append( tf.split(lambda_Lambda_update, K, 0) )			# split lambda_Lambda with each classes
eig_lambda_Lambda = []
eig_vec_lambda_Lambda = []
blocks_diag_lambda_Lambda = []
_lambda_Lambda_update = []
dets = []
off_diagonal = tf.ones([K, D, D]) - unit_matrix
epsilon = tf.constant(0.0001, shape=[K])
blocks_off_diagonal = []
for k in range(K):
    eig_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[0] )
    eig_vec_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[1] )
    for d in range(D):
        eig_lambda_Lambda[k] = tf.cond( tf.less_equal(eig_lambda_Lambda[k][d], 0.0), lambda: tf.abs(eig_lambda_Lambda[k]), lambda: eig_lambda_Lambda[k] )
    blocks_off_diagonal.append( tf.multiply( epsilon[k], off_diagonal[k] ) )
    blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() )
    _lambda_Lambda_update.append( tf.add( blocks_diag_lambda_Lambda[k], blocks_off_diagonal[k] ) )
    #_lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] )
    dets.append( tf.matrix_determinant(_lambda_Lambda_update[k]) )
lambda_Lambda_update = tf.assign( lambda_Lambda, tf.reshape( _lambda_Lambda_update , [K, D, D] ) )		# you should use tf.assign for updating parameter

print("* Care lambda_nu *")
_lambda_nu = []
dimension = tf.constant(D, dtype=tf.float32)
for k in range(K):
    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], dimension), lambda: dimension, lambda: lambda_nu_update[k] ) )
lambda_nu_update = tf.assign( lambda_nu, _lambda_nu )



# Session
cnt = 0
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#for epoch in range(100000):
#    #print( sess.run([log_q, grad_q_mu_tf, grad_q_muLambda_tf, grad_q_Lambda_tf, grad_q_nu_tf, lambda_mu_update, lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update]) )
#    print( sess.run([lambda_mu_update, lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update]) )
#    cnt = cnt + 1
#    print(cnt)
for epoch in range(1000):
    #print( sess.run( [unit_matrix, lambda_Lambda, lambda_Lambda_update] ) )
    #print( sess.run( [lambda_nu, lambda_nu_update] ) )
    sess.run( [lambda_mu_update, lambda_muLambda_update, lambda_Lambda_update, lambda_nu_update] )
    #print( sess.run( [element_wise_product_mu, element_wise_product_muLambda, element_wise_product_Lambda, element_wise_product_nu] ) )
    print( sess.run( element_wise_product_mu ) )
    #print( sess.run( sample_mean_mu ) )





