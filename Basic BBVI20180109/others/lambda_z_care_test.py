import tensorflow as tf
import numpy as np
import numpy.random as rand



def PGMethod(vector_subspace, element_numbers):
    normal_vector = tf.ones(element_numbers)
    vector_subspace_prime = tf.abs(vector_subspace)
    coefficient = tf.reduce_sum( tf.multiply(normal_vector, vector_subspace_prime) )
    norm = tf.norm(normal_vector)
    oriented_vector = tf.multiply( coefficient, tf.divide(normal_vector, norm) )
    element_sum = tf.reduce_sum( oriented_vector, axis=1 )
    vector_constrainted = tf.divide( oriented_vector, element_sum )
    
    return vector_constrainted


K = 2
D = 2
N = 3
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

init_lambda_z = tf.constant(np.array(rand.dirichlet(alpha=[0.1, 0.1], size=N), dtype=np.float32))
lambda_z = tf.Variable(init_lambda_z, dtype=tf.float32, trainable=True, name='lambda_z')
lambda_z_update = tf.convert_to_tensor([[0.2, 0.8], [1.5, -0.1], [0.2, 0.0]])
_lambda_z = []
pred_lambda_z0 = []
_lambda_z.append(tf.split(lambda_z_update, N, 0))
for n in range(N):
    pred_lambda_z0.append(
        tf.to_float(tf.logical_and(tf.greater(_lambda_z[0][n], tf.zeros([K])), tf.less(_lambda_z[0][n], tf.ones([K]))),
                name='pred_lambda_z0'))
    _lambda_z[0][n] = tf.cond(tf.logical_and(tf.less(tf.reduce_sum( pred_lambda_z0[n]), epsilon_pK), tf.greater(tf.reduce_sum( pred_lambda_z0[n]), epsilon_mK)), lambda: _lambda_z[0][n], lambda: PGMethod(_lambda_z[0][n], [1, K]), name='mid_lambda_z')    # K_f + 0.1, K_f - 0.1
_lambda_z_update = tf.concat( _lambda_z[0], 0, name='mid_lambda_z_update' )
lambda_z_cared = tf.assign(lambda_z, _lambda_z_update, name='lambda_z_cared')


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(lambda_z_cared))




