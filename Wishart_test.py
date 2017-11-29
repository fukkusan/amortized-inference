import tensorflow as tf



N = 10  	# number of data points
K = 3    	# number of components
D = 2     	# dimensionality of data
S = 4		# sample

#block_lambda_Lambda_1 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_Lambda_2 = tf.diag( tf.ones([1,D]) )[0]
#block_lambda_Lambda_3 = tf.diag( tf.ones([1,D]) )[0]
#unit_matrix = tf.reshape( tf.concat( [tf.concat( [block_lambda_Lambda_1, block_lambda_Lambda_2], 0 ), block_lambda_Lambda_3], 0 ), [K, D, D] )
unit_matrix = tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.ones([K, D, D]) ) ).to_dense()
#off_diag = tf.ones([K, D, D]) - unit_matrix
#V = 2 * unit_matrix + off_diag
lambda_Lambda = tf.Variable(unit_matrix , dtype=tf.float32, trainable=True, name='lambda_Lambda' )
#print(lambda_Lambda)
lambda_nu = tf.Variable( tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_nu' )
#print(lambda_nu)

update_counter = tf.Variable( tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter' )

rho = tf.Variable( tf.constant(0.1, shape=[1]), dtype=tf.float32, trainable=True )

#q_Lambda = tf.contrib.distributions.WishartFull( df=lambda_nu, scale=lambda_Lambda )
q_Lambda = tf.contrib.distributions.WishartCholesky( df=lambda_nu, scale=lambda_Lambda )


sample_q_Lambda = tf.Variable( tf.ones([S, K, D, D]) )
sample_q_Lambda_ass = tf.assign( sample_q_Lambda, q_Lambda.sample(sample_shape=[S]) )
#print(sample_q_Lambda_ass)
det_Lambda = tf.matrix_determinant(sample_q_Lambda_ass)

log_wishart = q_Lambda.log_prob( sample_q_Lambda_ass, name='log_wishart' )
log_q_Lambda = tf.reduce_sum( log_wishart, axis=1, name='log_q_Lambda' )
log_q = log_q_Lambda
#print(log_q)


grad_q_Lambda = []
grad_q_nu = []
for i in range(S):
    grad_q_Lambda.append( tf.gradients(log_q[i], lambda_Lambda) )
    grad_q_nu.append( tf.gradients(log_q[i], lambda_nu) )

grad_q_Lambda_tf = tf.convert_to_tensor(grad_q_Lambda)
grad_q_nu_tf = tf.convert_to_tensor(grad_q_nu)
#print(grad_q_Lambda_tf)

element_wise_product_Lambda = []
element_wise_product_nu = []
for j in range(S):
    element_wise_product_Lambda.append( tf.multiply(grad_q_Lambda_tf[j], log_q[j]) )
    element_wise_product_nu.append( tf.multiply(grad_q_nu_tf[j], log_q[j]) )
element_wise_product_Lambda_tf = tf.convert_to_tensor( element_wise_product_Lambda )
#print(element_wise_product_Lambda_tf)
sample_mean_Lambda = tf.reduce_mean( element_wise_product_Lambda_tf, axis = 0 )[0]
#print(sample_mean_Lambda)
sample_mean_nu = tf.reduce_mean( element_wise_product_nu, axis = 0 )[0]

#delta = tf.abs( tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part( tf.multiply(rho, sample_mean_Lambda) ) ).to_dense() )
delta_Lambda = tf.multiply(rho, sample_mean_Lambda)
lambda_Lambda_update = tf.assign(lambda_Lambda, tf.add( lambda_Lambda, delta_Lambda ) )
#print(lambda_Lambda_update)
#lambda_Lambda_update = tf.assign(lambda_Lambda, tf.add(lambda_Lambda, [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]) )
delta_nu = tf.multiply(rho, sample_mean_nu)
lambda_nu_update = tf.assign(lambda_nu, tf.add( lambda_nu, delta_nu ) )
#lambda_nu_update = tf.assign(lambda_nu, tf.add(lambda_nu, [1.0, 1.0, 1.0]) )


_lambda_Lambda = []
_lambda_Lambda.append( tf.split(lambda_Lambda_update, K, 0) )			# split lambda_Lambda with each classes
eig_lambda_Lambda = []
eig_vec_lambda_Lambda = []
blocks_diag_lambda_Lambda = []
_lambda_Lambda_update = []
dets = []
off_diagonal = tf.ones([K, D, D]) - unit_matrix
#epsilon = []
epsilon = tf.constant(0.001, shape=[K])
blocks_off_diagonal = []
for k in range(K):
    eig_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[0] )
    eig_vec_lambda_Lambda.append( tf.self_adjoint_eig( _lambda_Lambda[0][k][0] )[1] )
    for d in range(D):
        eig_lambda_Lambda[k] = tf.cond( tf.less_equal(eig_lambda_Lambda[k][d], 0.0), lambda: tf.abs(eig_lambda_Lambda[k]), lambda: eig_lambda_Lambda[k] )
    #epsilon.append( tf.sqrt( tf.sqrt( tf.reduce_prod(eig_lambda_Lambda[k]) ) ) )
    blocks_off_diagonal.append( tf.multiply( epsilon[k], off_diagonal[k] ) )
    blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() )
    _lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] + blocks_off_diagonal[k] )
    #_lambda_Lambda_update.append( blocks_diag_lambda_Lambda[k] )
    #_lambda_Lambda_update.append( tf.matmul( blocks_diag_lambda_Lambda[k], tf.matmul( eig_vec_lambda_Lambda[k], tf.matrix_inverse(eig_vec_lambda_Lambda[k]) ) ) )
    #blocks_diag_lambda_Lambda.append( tf.contrib.linalg.LinearOperatorDiag(eig_lambda_Lambda[k]).to_dense() + _lambda_Lambda[0][k][0] - tf.contrib.linalg.LinearOperatorDiag( tf.matrix_diag_part(_lambda_Lambda[0][k][0]) ).to_dense() )
    #dets.append( tf.matrix_determinant(_lambda_Lambda_update[k]) )
    dets.append( tf.matrix_determinant(_lambda_Lambda_update[k]) )
lambda_Lambda_update = tf.assign( lambda_Lambda, tf.reshape( _lambda_Lambda_update , [K, D, D] ) )		# you should use tf.assign for updating parameter

_lambda_nu = []
#_lambda_nu.append( tf.split(lambda_nu_update, K, 0) )
dimension = tf.constant(D, dtype=tf.float32)
df_Wishart_cond = tf.subtract( tf.to_float(D), tf.ones([1]) )
df_Wishart_new = tf.add( df_Wishart_cond, tf.ones([1]) )
#print(dimension)
for k in range(K):
    #_lambda_nu[0][k] = tf.cond( tf.less_equal(_lambda_nu[0][k][0], df_Wishart_cond[0]), lambda: df_Wishart_new[0], lambda: _lambda_nu[0][k] )
    #_lambda_nu[0][k] = tf.cond( tf.less_equal(_lambda_nu[0][k][0], tf.subtract(dimension, 1.0)), lambda: dimension , lambda: _lambda_nu[0][k] )
    #_lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], tf.to_float(D)-1.0), lambda: tf.to_float(D), lambda: lambda_nu_update[k] ) )
    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], dimension), lambda: dimension, lambda: lambda_nu_update[k] ) )
#lambda_nu_update = tf.assign( lambda_nu, tf.convert_to_tensor(_lambda_nu) )
#for k in range(K):
#    _lambda_nu.append( tf.cond( tf.less_equal(lambda_nu_update[k], tf.subtract(dimension, 1.0)), lambda: dimension, lambda: lambda_nu_update[k] ) )
#concated_lambda_nu = tf.concat(_lambda_nu[0], 0)
#print(concated_lambda_nu)
#lambda_nu_update = tf.assign( lambda_nu, concated_lambda_nu )
#_lambda_nu_tensor = tf.convert_to_tensor(_lambda_nu)
lambda_nu_update = tf.assign( lambda_nu, _lambda_nu )


# Session
sess = tf.Session()
init = tf.global_variables_initializer()
cnt = 0
sess.run(init)
for epoch in range(100000):
    #sess.run( lambda_Lambda_update )
    #print( sess.run( sample_mean_Lambda ) )
    print( sess.run( [lambda_Lambda_update, lambda_nu_update] ) )
    #print( sess.run( _lambda_nu[0] ) )
    #print( sess.run( concated_lambda_nu ) )
    #print( sess.run( lambda_nu_update ) )
    #print( sess.run( [det_Lambda, lambda_Lambda_update] ) )
    cnt = cnt + 1
    print(cnt)



