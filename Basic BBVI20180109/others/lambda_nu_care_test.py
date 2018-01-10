import tensorflow as tf
import numpy as np
import numpy.random as rand


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


_init_lambda_nu = np.array(rand.uniform(low=D-1+0.001, high=30.0, size=K), dtype=np.float32)
init_lambda_nu = tf.convert_to_tensor(_init_lambda_nu)
lambda_nu = tf.Variable(init_lambda_nu, dtype=tf.float32, trainable=True, name='lambda_nu')
lambda_nu_update = tf.convert_to_tensor([1.8, 1.0])
dimension = tf.constant(2.0, shape=[K], dtype=tf.float32, name='dimension')
pred_lambda_nu = tf.to_float(tf.greater(lambda_nu_update, tf.subtract(dimension, tf.ones([K]))))
_lambda_nu_cared = []
for k in range(K):
    _lambda_nu_cared.append(tf.cond(tf.logical_and(tf.greater(pred_lambda_nu[k], dimension[k]-1.0-epsilon),
                                              tf.less(pred_lambda_nu[k], dimension[k]-1.0+epsilon)),
                               lambda: lambda_nu_update[k], lambda: dimension[k], name='mid_lambda_nu_cared'))
lambda_nu_cared = tf.assign(lambda_nu, _lambda_nu_cared, name='lambda_nu_cared')


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([pred_lambda_nu, lambda_nu_cared]))


