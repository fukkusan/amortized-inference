import tensorflow as tf
import gc



N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample


unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
lambda_mu = tf.Variable( tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu' )
lambda_muLambda = tf.Variable( tf.ones([K]), dtype=tf.float32, trainable=True, name='lambda_muLambda' )

update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )
rho = tf.Variable( tf.constant(0.1, shape=[1]), dtype=tf.float32, trainable=True )

precision = []
for k in range(K):
    precision.append( tf.multiply( lambda_muLambda[k], unit_matrix[k] ) )
precision_tensor = tf.convert_to_tensor(precision)
covariance_q_mu = tf.matrix_inverse( precision_tensor )
del precision[:]
gc.collect()
q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=lambda_mu, covariance_matrix=covariance_q_mu )


sample_q_mu = tf.Variable( tf.zeros([S, K, D]) )
sample_q_mu_ass = tf.assign( sample_q_mu, q_mu.sample(sample_shape=[S]) )

log_gauss = q_mu.log_prob( sample_q_mu_ass, name='log_gauss' )
log_q_mu = tf.reduce_sum( log_gauss, axis=1, name='log_q_mu' )
log_q = log_q_mu

grad_q_mu = []
grad_q_muLambda = []
for i in range(S):
    grad_q_mu.append( tf.gradients(log_q[i], lambda_mu) )
    grad_q_muLambda.append( tf.gradients(log_q[i], lambda_muLambda) )

element_wise_product_mu = []
element_wise_product_muLambda = []
for j in range(S):
    element_wise_product_mu.append( tf.multiply(grad_q_mu[j], log_q[j]) )
    element_wise_product_muLambda.append( tf.multiply(grad_q_muLambda[j], log_q[j]) )
sample_mean_mu = tf.reduce_mean( element_wise_product_mu, axis = 0 )[0]
sample_mean_muLambda = tf.reduce_mean( element_wise_product_muLambda, axis = 0 )[0]

lambda_mu_update = tf.assign(lambda_mu, tf.add(lambda_mu, tf.multiply(rho, sample_mean_mu)) )
lambda_muLambda_update = tf.assign(lambda_muLambda, tf.add(lambda_muLambda, tf.multiply(rho, sample_mean_muLambda)) )


_lambda_muLambda = []
_lambda_muLambda.append( tf.split(lambda_muLambda_update, K, 0) )		# split lambda_muLambda with each classes
for k in range(K):
    _lambda_muLambda[0][k] = tf.cond( tf.less_equal(_lambda_muLambda[0][k][0], 0.0), lambda: tf.abs(_lambda_muLambda[0][k]), lambda: _lambda_muLambda[0][k] )
lambda_muLambda_update = tf.assign( lambda_muLambda, tf.concat(_lambda_muLambda[0], 0) )


# Session
sess = tf.Session()
init = tf.global_variables_initializer()
cnt = 0
sess.run(init)
for epoch in range(100000):
    print( sess.run( [lambda_mu_update, lambda_muLambda_update] ) )
    cnt = cnt + 1
    print(cnt)


