import tensorflow as tf


#A = [[3.0, 4.0], [4.0, 2.0]]
#A = [[3.0, 6.0], [6.0, 4.0]]
#A = [[5.0, 7.0], [7.0, 4.0]]
A = [[-1.78804302, 3.36202741], [3.36202741, 3.38826609]]
D = 2
epsilon = 0.0001

eig_val = tf.self_adjoint_eig(A)[0]
print(eig_val)
eig_vec = tf.self_adjoint_eig(A)[1]

pred_eig_val = []
pred_eig_val.append(tf.greater(eig_val, tf.zeros([D])))
print(pred_eig_val)

_eig_val_modify = []
for d in range(D):
    _eig_val_modify.append(tf.cond(tf.equal(pred_eig_val[0][d], True), lambda: eig_val[d], lambda: epsilon))
eig_val_modify = tf.convert_to_tensor(_eig_val_modify)

diag = tf.contrib.linalg.LinearOperatorDiag(eig_val_modify).to_dense()
B = tf.matmul(eig_vec, tf.matmul(diag, tf.matrix_inverse(eig_vec)))		# correct order
#B = tf.matmul(tf.matrix_inverse(eig_vec), tf.matmul(diag, eig_vec))
eig_val_B = tf.self_adjoint_eig(B)[0]


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.local_variables_initializer())

print(sess.run([diag, B, eig_val_B]))


