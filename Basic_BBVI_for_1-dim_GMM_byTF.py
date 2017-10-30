# Library
import numpy as np
import tensorflow as tf
import six
import numpy.random as nprand
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import math
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab




class BasicBBVI():
    # Constractor
    def __init__(self, num_data, num_cluster, vector_dim, num_sample, alpha_mean, alpha_var, gamma):
        # Parameters
        # Size parameters
        self.N = num_data
        self.K = num_cluster
        self.D = vector_dim
        self.S = num_sample
        
        # Input
        self.x = tf.placeholder(tf.float32, shape = [self.N, self.D])
        
        # Hyper parameters
        self.alpha_mean = tf.constant(alpha_mean, shape=[self.D, self.K], dtype=tf.float32)
        #self.alpha_mean = tf.constant(x_mean, shape=[self.D, self.K], dtype=tf.float32)
        self.alpha_var = tf.constant(alpha_var, shape=[1, 1], dtype=tf.float32)
        self.gamma = tf.constant(gamma, shape=[self.K], dtype=tf.float32)
        
        # Variational parameters
        self.lambda_pi = tf.Variable(tf.ones([self.K])/self.K, dtype=tf.float32, trainable=True, name='lambda_pi')
        self.lambda_mu = tf.Variable(tf.truncated_normal([self.D, self.K], mean = 0.0, stddev=2.0), dtype=tf.float32, trainable=True, name='lambda_mu')
        self.lambda_z = tf.Variable(tf.ones([self.N, self.K])/self.K, dtype=tf.float32, trainable=True, name='lambda_z')
        
        
        # Update count
        self.update_counter = 0
        
        
        # initialize distributions
        self.DistributionsUpdater()
        
        
        # Save previous variational parameters
        self.prev_lambda_pi = tf.ones(self.K)/self.K
        self.prev_lambda_mu = tf.truncated_normal([self.D, self.K], mean = 0.0, stddev=2.0)
        self.prev_lambda_z = tf.ones([self.N, self.K])/self.K
        
        
        
    # Update distributions
    def DistributionsUpdater(self):
        # Variational approximation model
        self.q_pi = tf.contrib.distributions.Dirichlet(self.lambda_pi)					# sample_shape=[self.K]
        self.q_mu = tf.contrib.distributions.Normal(self.lambda_mu, tf.ones(self.K))	# sample_shape=[self.D, self.K]
        self.q_z = tf.contrib.distributions.OneHotCategorical(self.lambda_z)					# sampe_shape=[self.N]
        
        
        # Generative model
        self.p_pi = tf.contrib.distributions.Dirichlet(self.gamma)						# sample_shape=[self.K]
        self.p_mu = tf.contrib.distributions.Normal(self.alpha_mean, self.alpha_var)	# sample_shape=[self.D, self.K]
        if self.update_counter > 0:
            self.pi_gene = self.q_pi.sample(sample_shape=[1])[0]
            self.mu_gene = self.q_mu.sample(sample_shape=[1])[0]
        else :
            self.pi_gene = self.p_pi.sample(sample_shape=[1])[0]
            self.mu_gene = self.p_mu.sample(sample_shape=[1])[0]
        self.p_z = tf.contrib.distributions.OneHotCategorical( self.pi_gene )					# sample_shape=[self.N, self.K]
        self.generative_gauss = tf.contrib.distributions.Normal(self.mu_gene, tf.ones(self.K))
        self.log_gene_gauss = self.generative_gauss.log_prob(self.generative_gauss.sample(sample_shape=[self.N]))
        #self.logpx = tf.reduce_sum( tf.multiply( tf.to_float( self.p_z.sample(sample_shape=[self.N, self.K]) ), self.log_gene_gauss ), axis=1 )		#sample_shape=[self.N, self.K]
        self.logpx = tf.reduce_sum( tf.multiply( tf.to_float( self.p_z.sample(sample_shape=[self.K]) ), self.log_gene_gauss ), axis=1 )		#sample_shape=[self.N, self.K]
        
        
        
        # Fit(Training) parameters
    def Fit(self, x_train):
        # Initiaize seesion
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        
        # Update
        vps_mu_1 = np.array(np.zeros(self.N//self.S), dtype=np.float32)
        vps_mu_2 = np.array(np.zeros(self.N//self.S), dtype=np.float32)
        vps_mu_3 = np.array(np.zeros(self.N//self.S), dtype=np.float32)
        df = pd.DataFrame(index=[], columns=['class1', 'class2', 'class3'])
        for epoch in range(self.N//self.S):
            vps = self.VariationalParametersUpdater(rho = 0.1)
            vps = self.Care()
            variational_parameters = sess.run(vps, feed_dict = {
                self.x: x_train
            })
        # for debug
            print(variational_parameters[1])		# variational_parameters[1] is self.lambda_mu
            plt.figure(1)
            vps_mu_1[epoch] = variational_parameters[1][0][0]
            vps_mu_2[epoch] = variational_parameters[1][0][1]
            vps_mu_3[epoch] = variational_parameters[1][0][2]
            q_mu_0 = np.array(np.zeros(20), dtype=np.float32)
            q_mu_1 = np.array(np.zeros(20), dtype=np.float32)
            q_mu_2 = np.array(np.zeros(20), dtype=np.float32)
            x = np.array(np.zeros(20), dtype=np.float32)
            norm1 = mlab.normpdf(x_train, 0.0, 1.0)
            norm2 = mlab.normpdf(x_train, 2.0, 1.0)
            norm3 = mlab.normpdf(x_train, -2.0, 1.0)
            log_likelihood_p = np.array(np.zeros(10), dtype=np.float32)
            log_q = np.array(np.zeros(10), dtype=np.float32)
            epochs = np.array(np.zeros(10), dtype=np.float32)
            epochs[epoch] = epoch
            for i in range(20):
                x_element = -5.0 + i/10*5
                x[i] = x_element
                q_mu_0[i] =  self.q_mu.prob( x_element ).eval(session=sess)[0][0] 
                q_mu_1[i] =  self.q_mu.prob( x_element ).eval(session=sess)[0][1] 
                q_mu_2[i] =  self.q_mu.prob( x_element ).eval(session=sess)[0][2] 
            log_likelihood_p[epoch] = np.mean(self.log_p.eval(session=sess))
            log_q[epoch] = np.mean(self.log_p.eval(session=sess))
            plt.plot([-2.0, -2.0, -2.0], [-1.0, 0.0, 1.0], linestyle="-.", color="b")
            plt.plot([0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], linestyle="-.", color="g")
            plt.plot([2.0, 2.0, 2.0], [-1.0, 0.0, 1.0], linestyle="-.", color="r")
            plt.plot(x, q_mu_0, color="b", label="q(mu_1|lambda_mu_1)")
            plt.plot(x, q_mu_1, color="g", label="q(mu_2|lambda_mu_2)")
            plt.plot(x, q_mu_2, color="r", label="q(mu_3|lambda_mu_3)")
            plt.scatter(x_train, norm1, color="g")
            plt.scatter(x_train, norm2, color="r")
            plt.scatter(x_train, norm3, color="b")
            plt.xlim([-5.0, 5.0])
            plt.ylim([0.0, 1.0])
            plt.title("q(mu|lambda_mu)")
            plt.legend()
            plt.show()
            del q_mu_0
            del q_mu_1
            del q_mu_2
            del x
            plt.clf()
            plt.close()
            se = pd.Series([vps_mu_1[epoch], vps_mu_2[epoch], vps_mu_3[epoch]], index=df.columns)
            df = df.append(se, ignore_index=True)
        plt.plot(epochs, log_likelihood_p, label="log_p")
        plt.plot(epochs, log_q, label="log_q")
        plt.title("log_p and log_q")
        plt.legend()
        plt.show()
        df.to_csv("output_lambda_mu.csv", index=False)
    
    
    
    # Inference variational parameters
    # Update function for variational parameters
    def VariationalParametersUpdater(self, rho):
        s = 0
        # Sampling and Logarithmic distributions
        self.log_p_x = list( np.ones(self.S, dtype=np.float32) )
        self.log_p_z = list( np.ones(self.S, dtype=np.float32) )
        self.log_p_mu = list( np.ones(self.S, dtype=np.float32) )
        self.log_p_pi = list( np.ones(self.S, dtype=np.float32) )
        self.log_q_pi = list( np.ones(self.S, dtype=np.float32) )
        self.log_q_mu = list( np.ones(self.S, dtype=np.float32) )
        self.log_q_z = list( np.ones(self.S, dtype=np.float32) )
        
        
        while(s < self.S):
            # Update distributions
            if(self.update_counter>0):
                self.DistributionsUpdater()
            
            self.log_p_x[s] = tf.reduce_sum( self.logpx )
            self.log_p_mu[s] = tf.reduce_sum( self.p_mu.log_prob(self.p_mu.sample(sample_shape=[1]))[0][0] )
            self.log_p_pi[s] = self.p_pi.log_prob(self.p_pi.sample(sample_shape=[1]))[0]
            
            self.log_dirichlet = self.q_pi.log_prob(self.q_pi.sample(sample_shape=[1]))[0]
            self.log_categorical = self.q_z.log_prob(self.q_z.sample(sample_shape=[1]))[0]
            self.log_gauss = self.q_mu.log_prob(self.q_mu.sample(sample_shape=[1]))[0]
            self.log_q_pi[s] = self.log_dirichlet
            self.log_q_mu[s] = tf.reduce_sum( self.log_gauss )
            self.log_q_z[s] = tf.reduce_sum( self.log_categorical )
            
            s = s + 1
        
        
        
        self.log_p = tf.add( tf.add( tf.add( tf.convert_to_tensor(self.log_p_x), tf.convert_to_tensor(self.log_p_z) ), tf.convert_to_tensor(self.log_p_pi) ), tf.convert_to_tensor(self.log_p_mu) ) 
        self.log_q = tf.add( tf.add( tf.convert_to_tensor(self.log_q_z),  tf.convert_to_tensor(self.log_q_mu) ), tf.convert_to_tensor(self.log_q_pi) ) 
        self.log_loss = tf.subtract( self.log_p, self.log_q )
        
        
        # Gradient
        self.grad_q_pi = tf.gradients(self.log_q, self.lambda_pi)
        self.grad_q_mu = tf.gradients(self.log_q, self.lambda_mu)
        self.grad_q_z = tf.gradients(self.log_q, self.lambda_z)
        
        
        # Sample mean(Montecarlo Approximation)
        self.element_wise_product_pi = []
        self.element_wise_product_mu = []
        self.element_wise_product_z = []
        for j in range(self.S):
            self.element_wise_product_pi.append(tf.multiply(self.grad_q_pi, self.log_loss[j]))
            self.element_wise_product_mu.append(tf.multiply(self.grad_q_mu, self.log_loss[j]))
            self.element_wise_product_z.append(tf.multiply(self.grad_q_z, self.log_loss[j]))
        self.sample_mean_pi = tf.reduce_mean( self.element_wise_product_pi, axis = 0 )[0]
        self.sample_mean_mu = tf.reduce_mean( self.element_wise_product_mu, axis = 0 )[0]
        self.sample_mean_z = tf.reduce_mean( self.element_wise_product_z, axis = 0 )[0]
        
        
        # Update variational parameters
        self.lambda_pi = tf.add(self.lambda_pi, tf.multiply(rho, self.sample_mean_pi))
        self.lambda_mu = tf.add(self.lambda_mu, tf.multiply(rho, self.sample_mean_mu))
        self.lambda_z = tf.add(self.lambda_z, tf.multiply(rho, self.sample_mean_z))
        
        
        self.update_counter = self.update_counter + 1
        
        
        return [self.lambda_pi, self.lambda_mu, self.lambda_z]
    
    
    
    # Care Values
    def Care(self):
        lambda_pi = []
        lambda_pi.append( tf.split(self.lambda_pi, self.K, 0) )
        k=0
        while(k < self.K):
            lambda_pi[0][k] = tf.cond( tf.less_equal( lambda_pi[0][k][0], 0.0 ), lambda: tf.abs( lambda_pi[0][k] ), lambda: lambda_pi[0][k] )
            k = k + 1
        if(k == self.K):
            self.lambda_pi = tf.concat(lambda_pi[0], 0)
        
        
        lambda_z = []
        lambda_z.append( tf.split(self.lambda_z, self.N, 0) )
        n=0
        while(n < self.N):
            k=0
            while(k < self.K):
                lambda_z[0][n] = tf.cond( tf.less_equal( lambda_z[0][n][0][k], 0.0 ), lambda: tf.abs(lambda_z[0][n]), lambda: lambda_z[0][n] )
                k = k + 1
            #lambda_z[0][n] = tf.abs( lambda_z[0][n] )
            lambda_z[0][n] = tf.cond( tf.not_equal( tf.reduce_sum( lambda_z[0][n] ), 1.0 ), lambda: self.ConstraintMethod( lambda_z[0][n] ), lambda: lambda_z[0][n] )
            n = n + 1
        if(n == self.N):
            self.lambda_z = tf.concat(lambda_z[0], 0)
        
        
        # Deal with nan and inf
        for i in range(self.K):
            self.lambda_pi = tf.cond( tf.equal( tf.is_inf(self.lambda_pi)[i], True ), lambda: self.prev_lambda_pi, lambda: self.lambda_pi )
            self.lambda_pi = tf.cond( tf.equal( tf.is_nan(self.lambda_pi)[i], True ), lambda: self.prev_lambda_pi, lambda: self.lambda_pi )
            self.lambda_mu = tf.cond( tf.equal( tf.is_inf(self.lambda_mu)[0][i], True ), lambda: self.prev_lambda_mu, lambda: self.lambda_mu )
            self.lambda_mu = tf.cond( tf.equal( tf.is_nan(self.lambda_mu)[0][i], True ), lambda: self.prev_lambda_mu, lambda: self.lambda_mu )
            self.lambda_z = tf.cond( tf.equal( tf.is_inf(self.lambda_z)[0][i], True ), lambda: self.prev_lambda_z, lambda: self.lambda_z )
            self.lambda_z = tf.cond( tf.equal( tf.is_nan(self.lambda_z)[0][i], True ), lambda: self.prev_lambda_z, lambda: self.lambda_z )
        
        
        # Previous lambda
        self.prev_lambda_pi = self.lambda_pi
        self.prev_lambda_mu = self.lambda_mu
        self.prev_lambda_z = self.lambda_z
        
        
        return [self.lambda_pi, self.lambda_mu, self.lambda_z]
    
    
    
    # Constraint Method
    def ConstraintMethod(self, vector_subspace):
        element_sum = tf.reduce_sum( vector_subspace )
        vector_constrainted = tf.divide( vector_subspace, element_sum )
        
        
        return vector_constrainted
    
    
    
# Main
if __name__ == '__main__':
    N = 100  	# number of data points
    K = 3    	# number of components
    D = 1     	# dimensionality of data
    S = 10		# sample
    alpha = 0.0
    beta = 2.0
    gamma = 0.1
    
    sample_x1 = nprand.normal(alpha, beta, int(N*0.7))		# Mixture ratio pi_1=0.7
    sample_x2 = nprand.normal(alpha+2.0, beta, int(N*0.2))	# Mixture ratio pi_2=0.2
    sample_x3 = nprand.normal(alpha-2.0, beta, int(N*0.1))	# Mixture ratio pi_3=0.1
    x = np.reshape( np.concatenate( [np.concatenate([sample_x1, sample_x2]), sample_x3] ), (N, D) )
    
    x_mean = np.mean(x)
    
    sfg = BasicBBVI(N, K, D, S, x_mean, beta, gamma)
    sfg.Fit(x)
    



