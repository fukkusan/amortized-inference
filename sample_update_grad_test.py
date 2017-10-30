import tensorflow as tf


# Initialize
K=1
lambda_mu = tf.Variable(tf.zeros(K), dtype=tf.float32)
#p_mu = tf.contrib.distributions.Normal(lambda_mu, tf.ones(K))
p_mu = tf.contrib.distributions.Uniform(low=lambda_mu, high=5.0)


# Sampling
#norm_lambda = tf.norm(lambda_mu)
#delta = tf.add(norm_lambda, tf.ones(1))
#pdf_p_mu = p_mu.prob(2.0)			# do not work update
sample = tf.Variable(tf.zeros([1, 1]))
sample = tf.assign( sample, p_mu.sample(sample_shape=[1]) )
delta = tf.gradients( p_mu.log_prob(sample)[0], lambda_mu )[0]
#delta = p_mu.log_prob(sample)[0]
#log_p_mu = tf.reduce_sum( p_mu.log_prob(sample), axis = 1 )


#Update parameter
#_lambda_mu = tf.add(lambda_mu, log_p_mu)
_lambda_mu = tf.add(lambda_mu, delta)
#_lambda_mu = tf.add(lambda_mu, tf.ones(1))
lambda_mu = tf.assign(lambda_mu, _lambda_mu)


#Care parameter
#lambda_mu = tf.cond( tf.less(lambda_mu[0], 0.4), lambda: 0.5, lambda: lambda_mu )


#Update distribution
#check = tf.cond( tf.less(lambda_mu[0], 0.4), lambda: True, lambda: False )
#if(check==True):
#    lambda_mu = tf.assign(lambda_mu, 0.5)		# do not work if sentence
p_mu = tf.contrib.distributions.Uniform(low=lambda_mu, high=5.0)
#delta = p_mu.log_prob(sample)[0]
delta = tf.gradients( p_mu.log_prob(sample)[0], lambda_mu )[0]
pdf_p_mu = p_mu.prob(2.0)



# Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(3):
    result = sess.run(delta)
    print(result)




