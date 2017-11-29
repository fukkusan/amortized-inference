import tensorflow as tf
import numpy as np
import gc
from tensorflow.python import debug



N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample

unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
lambda_Lambda = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )
#lambda_Lambda_update = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda_update' )
#lambda_nu_update = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu_update' )

update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )

rho = tf.Variable( tf.constant(0.1, shape=[1]), dtype=tf.float32, trainable=True )

q_Lambda = tf.contrib.distributions.WishartCholesky( df=lambda_nu, scale=lambda_Lambda )


sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )
#det_Lambda = tf.matrix_determinant(sample_q_Lambda_ass)

#log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
#watch_log_wishart = tfdbg.add_debug_tensor_wacth(run_options=FULLTRACE, node_name='log_wishart')
#log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
log_q_Lambda = tf.reduce_sum( q_Lambda.log_prob( sample_q_Lambda_ass ), axis=1, name='log_q_Lambda' )
log_q = log_q_Lambda

grad_q_Lambda = []
grad_q_nu = []
for i in range(S):
    grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda) )
    grad_q_nu.append( tf.gradients(log_q[i], lambda_nu) )
grad_q_Lambda_tf = tf.convert_to_tensor(grad_q_Lambda)
grad_q_nu_tf = tf.convert_to_tensor(grad_q_nu)
del grad_q_Lambda[:]
del grad_q_nu[:]
gc.collect()

element_wise_product_Lambda = []
element_wise_product_nu = []
for j in range(S):
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda_tf[j], log_q[j]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu_tf[j], log_q[j]) )
element_wise_product_Lambda_tf = tf.convert_to_tensor( element_wise_product_Lambda )
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda_tf, axis = 0 )[0]
sample_mean_nu = tf.reduce_mean( element_wise_product_nu, axis = 0 )[0]
del element_wise_product_Lambda[:]
del element_wise_product_nu[:]
gc.collect()


delta_Lambda = tf.multiply(rho, sample_mean_Lambda)
lambda_Lambda_update = tf.assign(lambda_Lambda, tf.add( lambda_Lambda, delta_Lambda ) )
#lambda_Lambda_update = tf.add( lambda_Lambda, delta_Lambda )
delta_nu = tf.multiply(rho, sample_mean_nu)
lambda_nu_update = tf.assign(lambda_nu, tf.add( lambda_nu, delta_nu ) )
#lambda_nu_update = tf.add( lambda_nu, delta_nu )


_lambda_Lambda = []
_lambda_Lambda.append( tf.split(lambda_Lambda, K, 0) )			# split lambda_Lambda with each classes
eig_lambda_Lambda = []
eig_vec_lambda_Lambda = []
blocks_diag_lambda_Lambda = []
_lambda_Lambda_update = []
off_diagonal = tf.ones([K, D, D]) - unit_matrix
#assert_sym = []
for k in range(K):
    offDiag_lambda_Lambda = tf.multiply( tf.slice( _lambda_Lambda[0][k][0], [1, 0], [1, 1] ), off_diagonal[k] )
    _lambda_Lambda[0][k] = tf.add( tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( _lambda_Lambda[0][k] ) ).to_dense(), offDiag_lambda_Lambda )
    #symmetry = tf.subtract( _lambda_Lambda[0][k][0][0][1], _lambda_Lambda[0][k][0][1][0] )
    #assert_sym.append( tf.assert_equal( symmetry, tf.zeros([1]) ) )				# Is any problem?
    eig_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[0] )
    eig_vec_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[1] )
    for d in range(D):
        eig_lambda_Lambda[k] = tf.cond( tf.less_equal(eig_lambda_Lambda[k][d], 0.0), lambda: tf.abs(eig_lambda_Lambda[k]), lambda: eig_lambda_Lambda[k] )
    blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() )
    _lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] )
lambda_Lambda_cared = tf.assign( lambda_Lambda, tf.reshape( _lambda_Lambda_update , [K, D, D] ) )		# you should use tf.assign for updating parameter
del _lambda_Lambda[:]
del eig_lambda_Lambda[:]
del eig_vec_lambda_Lambda[:]
del blocks_diag_lambda_Lambda[:]
del _lambda_Lambda_update[:]
gc.collect()

_lambda_nu = []
dimension = tf.constant(D, dtype=tf.float32)
for k in range(K):
    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu[k], dimension), lambda: dimension, lambda: lambda_nu_update[k] ) )
lambda_nu_cared = tf.assign( lambda_nu, _lambda_nu )
del _lambda_nu[:]
gc.collect()



# Session
sess = tf.Session()
init = tf.global_variables_initializer()
cnt = 0
sess.run(init)
sess = debug.LocalCLIDebugWrapperSession(sess)
for epoch in range(700000):
    #sess.run( lambda_Lambda_update )
    #print( sess.run( sample_mean_Lambda ) )
    print( sess.run( [lambda_Lambda_cared, lambda_nu_cared] ) )
    #print( sess.run( [lambda_Lambda_update, lambda_Lambda_cared, lambda_nu_update, lambda_nu_cared] ) )
    #print( sess.run( [lambda_Lambda, lambda_Lambda_update, lambda_nu, lambda_nu_update] ) )
    #print( sess.run( [symmetry, assert_sym, lambda_Lambda_update, lambda_nu_update] ) )
    #print( sess.run( _lambda_nu[0] ) )
    #print( sess.run( concated_lambda_nu ) )
    #print( sess.run( lambda_nu_update ) )
    #print( sess.run( [det_Lambda, lambda_Lambda_update] ) )
    cnt = cnt + 1
    print(cnt)



