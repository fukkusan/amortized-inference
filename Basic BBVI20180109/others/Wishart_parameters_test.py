import tensorflow as tf
import Gaussian_plot as gp
from matplotlib import pyplot as plt



num_epochs = 1


Lambda = tf.constant([[0.9, 0.0], [0.0, 0.9]])
nu = tf.constant(9.0)
mu = tf.constant([0.0, 0.0])

q_Lambda = tf.contrib.distributions.WishartFull(df=nu, scale=Lambda, name='q_Lambda')
sample_q_Lambda = q_Lambda.sample(sample_shape=[6])
expectation_sample_q_Lambda = tf.reduce_mean(sample_q_Lambda, axis=0)
print(expectation_sample_q_Lambda)
covariance = tf.matrix_inverse(sample_q_Lambda)
covariance_expectation = tf.matrix_inverse(expectation_sample_q_Lambda)
p_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance, name='p_gauss')
sample_p_gauss = p_gauss.sample(sample_shape=[1])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(tf.local_variables_initializer())


fig = plt.figure()
ax = plt.axes()
ax.set_xlim([-5.0, 5.0])
ax.set_ylim([-5.0, 5.0])
for epoch in range(num_epochs):
    result = sess.run([mu, covariance, covariance_expectation, sample_q_Lambda])
    
    print("mu")
    print(result[0])
    print("covariance_0")
    print(result[1][0])
    print("covariance_1")
    print(result[1][1])
    print("covariance_2")
    print(result[1][2])
    print("covariance_3")
    print(result[1][3])
    print("covariance_4")
    print(result[1][4])
    print("covariance_5")
    print(result[1][5])
    print("covariance_expectation")
    print(result[2])
    
    gp.plot_mean_variance(ax, result[0], result[1][0], 'cyan', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[1][1], 'midnightblue', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[1][2], 'salmon', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[1][3], 'tomato', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[1][4], 'lime', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[1][5], 'forestgreen', scale=1.0)
    gp.plot_mean_variance(ax, result[0], result[2], 'k', scale=1.0)
plt.show()



