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


init_lambda_pi = tf.constant(np.array(rand.gamma(shape=0.27, size=K), dtype=np.float32))
lambda_pi = tf.Variable(init_lambda_pi, dtype=tf.float32, trainable=True, name='lambda_pi')
lambda_pi_update = tf.convert_to_tensor([0.5, -0.1])
pred_lambda_pi = tf.to_float(tf.greater(lambda_pi_update, tf.zeros([K])), name='pred_lambda_pi')
#_lambda_pi_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_pi), epsilon_mK), tf.less(tf.reduce_sum(pred_lambda_pi), epsilon_pK)), lambda: lambda_pi_update, lambda: tf.abs(tf.multiply(0.5, lambda_pi_update)), name='mid_lambda_pi_cared')
_lambda_pi_cared = []
for k in range(K):
    _lambda_pi_cared.append(tf.cond(tf.logical_and(tf.greater(pred_lambda_pi[k], epsilon_minus), tf.less(pred_lambda_pi[k], epsilon_plus)), lambda: lambda_pi_update[k], lambda: tf.abs(tf.multiply(0.5, lambda_pi_update[k])), name='mid_lambda_pi_cared'))
_lambda_pi_cared_tf = tf.convert_to_tensor(_lambda_pi_cared)
lambda_pi_cared = tf.assign(lambda_pi, _lambda_pi_cared_tf, name='lambda_pi_cared')


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([pred_lambda_pi, lambda_pi_cared]))

