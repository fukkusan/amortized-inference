import tensorflow as tf


K = 2
D = 2
epsilon = tf.constant(0.0001)    # 10^{-4}
K_f = tf.to_float(K)
D_f = tf.to_float(D)
KD_f = tf.multiply(K_f, D_f)
epsilon_plus = tf.add(1.0, epsilon)
epsilon_minus = tf.subtract(1.0, epsilon)
epsilon_pK = tf.add(K_f, epsilon)
epsilon_mK = tf.subtract(K_f, epsilon)
epsilon_pKD = tf.add(KD_f, epsilon)
epsilon_mKD = tf.subtract(KD_f, epsilon)
det_lower_limit = tf.constant(0.0001, name='det_lower_limit')
det_care_diag_epsilon = tf.constant([[[0.001, 0.0], [0.0, 0.001]], [[0.001, 0.0], [0.0, 0.001]]], dtype=tf.float32, name='det_care_diag_epsilon')
det_care_count4 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count4')

unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([K, D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([K, D, D]), unit_matrices)

lambda_muLambda = tf.Variable(unit_matrices, dtype=tf.float32, trainable=True, name='lambda_muLambda')
lambda_muLambda_update = tf.convert_to_tensor([[[3.0, 4.0], [4.0, 2.0]], [[3.0, 6.0], [6.0, 4.0]]])


#lambda_muLambda_update_modify = tf.multiply(lambda_muLambda_update, unit_matrices)
_off_diagonal_lambda_muLambda = []
for k in range(K):
    _off_diagonal_lambda_muLambda.append(tf.multiply(lambda_muLambda_update[k][0][1], off_diagonal[k]))
off_diagonal_lambda_Lambda = tf.convert_to_tensor(_off_diagonal_lambda_muLambda)
dig_lambda_muLambda_update = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(lambda_muLambda_update)).to_dense()
lambda_muLambda_update_modify = tf.add(dig_lambda_muLambda_update, off_diagonal_lambda_Lambda)
eig_val_lambda_muLambda = tf.self_adjoint_eigvals(lambda_muLambda_update, name='eig_val_lambda_muLambda')
eig_val_muLambda_minimum = tf.constant(0.001, shape=[K, D], dtype=tf.float32)
eig_val_lambda_muLambda_modify = tf.maximum(eig_val_muLambda_minimum, eig_val_lambda_muLambda)
diagonalized_muLambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_muLambda_modify, name='diagonalized_muLambda').to_dense()
pred_lambda_muLambda = tf.to_float(tf.greater(eig_val_lambda_muLambda, tf.zeros([K, D])))
_lambda_muLambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_muLambda), epsilon_mKD), tf.less(tf.reduce_sum(pred_lambda_muLambda), epsilon_pKD)), lambda: lambda_muLambda_update, lambda: diagonalized_muLambda)		# Sum of pred_lambda_muLambda is K*D if all eigenvalues are positive.
lambda_muLambda_cared = tf.assign(lambda_muLambda, _lambda_muLambda_cared, name='lambda_muLambda_cared')


## care det = 0
#det_lambda_muLambda_cared = tf.matrix_determinant(_lambda_muLambda_cared)
#det_min4 = tf.reduce_min(tf.abs(det_lambda_muLambda_cared))
#det_care_modified4 = tf.add(_lambda_muLambda_cared, det_care_diag_epsilon)
#det_care_new4 = tf.cond(tf.less(det_min4, det_lower_limit), lambda: det_care_modified4,
#                        lambda: _lambda_muLambda_cared)
#lambda_muLambda_cared = tf.assign(lambda_muLambda, det_care_new4, name='lambda_muLambda_cared')
#det_care_c4 = tf.cond(tf.less(det_min4, det_lower_limit), lambda: tf.add(det_care_count4, 1),
#                      lambda: det_care_count4)
#det_care_count4_ass = tf.assign(det_care_count4, det_care_c4)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([lambda_muLambda_cared, eig_val_lambda_muLambda, eig_val_lambda_muLambda_modify]))






