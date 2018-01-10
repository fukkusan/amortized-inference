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
det_care_count5 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count5')

unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([K, D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([K, D, D]), unit_matrices)

lambda_Lambda = tf.Variable(unit_matrices, dtype=tf.float32, trainable=True, name='lambda_Lambda')
lambda_Lambda_update = tf.convert_to_tensor([[[3.0, 4.0], [4.0, 2.0]], [[3.0, 6.0], [6.0, 4.0]]])


_off_diagonal_lambda_Lambda = []
for k in range(K):
    _off_diagonal_lambda_Lambda.append(tf.multiply(lambda_Lambda_update[k][0][1], off_diagonal[k]))
off_diagonal_lambda_Lambda = tf.convert_to_tensor(_off_diagonal_lambda_Lambda)
dig_lambdaLambda_update = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(lambda_Lambda_update)).to_dense()
lambda_Lambda_update_modify = tf.add(dig_lambdaLambda_update, off_diagonal_lambda_Lambda)
eig_val_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_val_lambda_Lambda')[0]
eig_vec_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_vec_lambda_Lambda')[1]
eig_val_Lambda_minimum = tf.constant(0.001, shape=[K, D], dtype=tf.float32)
eig_val_lambda_Lambda_modify = tf.maximum(eig_val_Lambda_minimum, eig_val_lambda_Lambda)
diagonalized_Lambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_Lambda_modify, name='diagonalized_Lambda').to_dense()
lambda_Lambda_new = tf.matmul(eig_vec_lambda_Lambda, tf.matmul(diagonalized_Lambda, tf.matrix_inverse(eig_vec_lambda_Lambda)), name='lambda_Lambda_new')
pred_lambda_Lambda = tf.to_float(tf.greater(eig_val_lambda_Lambda, tf.zeros([K, D])),
                             name='pred_lambda_Lambda')  # The condition of tf.cond must compare scalars but this tf.less_equal gives boolean tensor with shape [K, D].
_lambda_Lambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_Lambda), epsilon_mKD), tf.less(tf.reduce_sum(pred_lambda_Lambda), epsilon_pKD)), lambda: lambda_Lambda_update, lambda: lambda_Lambda_new)		# Sum of pred_lambda_Lambda is K*D if all eigenvalues are positive.
_off_diagonal_lambda_Lambda_cared = []
for k in range(K):
    _off_diagonal_lambda_Lambda_cared.append(tf.multiply(_lambda_Lambda_cared[k][0][1], off_diagonal[k]))
off_diagonal_lambda_Lambda_cared = tf.convert_to_tensor(_off_diagonal_lambda_Lambda_cared)
dig_lambda_Lambda_cared = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(_lambda_Lambda_cared)).to_dense()
_lambda_Lambda_cared_modify = tf.add(dig_lambda_Lambda_cared, off_diagonal_lambda_Lambda_cared)
lambda_Lambda_cared = tf.assign(lambda_Lambda, _lambda_Lambda_cared_modify, name='lambda_Lambda_cared')
eig_val_lambda_Lambda_cared = tf.self_adjoint_eigvals(lambda_Lambda_cared)

## care det = 0
#det_lambda_Lambda_cared = tf.matrix_determinant(_lambda_Lambda_cared_modify)
#det_min5 = tf.reduce_min(tf.abs(det_lambda_Lambda_cared))
#det_care_modified5 = tf.add(_lambda_Lambda_cared_modify, det_care_diag_epsilon)
#det_care_new5 = tf.cond(tf.less(det_min5, det_lower_limit), lambda: det_care_modified5,
#                        lambda: _lambda_Lambda_cared_modify)
#lambda_Lambda_cared = tf.assign(lambda_Lambda, det_care_new5, name='lambda_Lambda_cared')
#det_care_c5 = tf.cond(tf.less(det_min5, det_lower_limit), lambda: tf.add(det_care_count5, 1),
#                      lambda: det_care_count5)
#det_care_count5_ass = tf.assign(det_care_count5, det_care_c5)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([lambda_Lambda_cared, eig_val_lambda_Lambda_cared, eig_val_lambda_Lambda, eig_val_lambda_Lambda_modify]))


