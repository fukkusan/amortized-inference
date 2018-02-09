import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.random as rand
import scipy.stats as static
import Input_data_for_2_dim_GMM_K2 as inp
import calc_ELBO as cal_elbo
import gc




# Projected Gradient Method
def PGMethod(vector_subspace, element_numbers):
    normal_vector = tf.ones(element_numbers)
    vector_subspace_prime = tf.abs(vector_subspace)
    coefficient = tf.reduce_sum( tf.multiply(normal_vector, vector_subspace_prime) )
    norm = tf.norm(normal_vector)
    oriented_vector = tf.multiply( coefficient, tf.divide(normal_vector, norm) )
    element_sum = tf.reduce_sum( oriented_vector, axis=1 )
    vector_constrainted = tf.divide( oriented_vector, element_sum )
    #vector_constrainted = oriented_vector
    
    
    return vector_constrainted




# * Initialize *
print("* Initialize *")
# Constants
N = 1000  	# number of data points
K = 2    	# number of components
D = 2     	# dimensionality of data
#S = 50		# sample
S = 100
KS = K * S
#_alpha = 0.0
_beta = tf.constant(2.0, shape=[K, D, D])
_gamma = 1.0
#num_epochs = 30
#num_samples = 50
num_epochs = 1000
num_samples = S
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

# care det = 0
det_lower_limit = tf.constant(0.0001, name='det_lower_limit')
det_care_diag_epsilon = tf.constant([[[0.001, 0.0], [0.0, 0.001]], [[0.001, 0.0], [0.0, 0.001]]], dtype=tf.float32, name='det_care_diag_epsilon')
det_care_count1 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count1')
det_care_count2 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count2')
det_care_count3 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count3')
det_care_count4 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count4')
det_care_count5 = tf.Variable(tf.constant(0), tf.int32, name='det_care_count5')

# Observable
input_data = inp.Input()
x = input_data[0]
x_mean = input_data[1]
print("Input data")
print(x)

# Input placeholder
X = tf.placeholder(tf.float32, shape=[N, D], name='observable')


# Parameters
# Hyper parameters
unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([K, D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([K, D, D]), unit_matrices)
sample_unit = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([S, K, D, D])), name='sample_unit').to_dense()
hyper_alpha_mean = tf.constant([[3.2, 0.2], [-0.2, 3.2]], dtype=tf.float32, name='hyper_alpha_mean')
hyper_coe_alpha_var = tf.multiply(sample_unit, _beta, name='hyper_coe_alpha_var')
hyper_gamma = tf.constant([0.3, 4.0], shape=[K], dtype=tf.float32, name='hyper_gamma')
hyper_V = tf.constant([[[2.0, 0.8], [-0.3, 4.0]], [[5.0, 0.0], [0.0, 5.0]]])
hyper_nu = tf.constant(2.05, shape=[K])
init_lambda_mu = tf.constant([[2.8, 0.2], [0.2, 2.8]], dtype=tf.float32)
init_lambda_muLambda = unit_matrices
init_lambda_Lambda = tf.constant([[[0.6, 0.0], [0.0, 0.6]], [[0.6, 0.0], [0.0, 0.6]]], dtype=tf.float32)
init_lambda_nu = tf.constant(5.0, shape=[K], dtype=tf.float32)
#init_lambda_z = tf.divide(tf.ones([N, K]), K)
#_init_lambda_mu = [tf.convert_to_tensor(np.array(rand.multivariate_normal(mean=[3.0, 0.0], cov=[[0.1, 0.0], [0.0, 0.1]]), dtype=np.float32)), tf.convert_to_tensor(np.array(rand.multivariate_normal(mean=[0.0, 3.0], cov=[[0.1, 0.0], [0.0, 0.1]]), dtype=np.float32))]
#init_lambda_mu = tf.convert_to_tensor(_init_lambda_mu)
#_init_lambda_Lambda = [tf.convert_to_tensor(np.array(static.wishart.rvs(df=20.0, scale=[[0.5, 0.0], [0.0, 0.5]]), dtype=np.float32)), tf.convert_to_tensor(np.array(static.wishart.rvs(df=20.0, scale=[[0.5, 0.0], [0.0, 0.5]]), dtype=np.float32))]
#init_lambda_Lambda = tf.convert_to_tensor(_init_lambda_Lambda)
#_init_lambda_nu = np.array(rand.uniform(low=D-1+0.001, high=30.0, size=K), dtype=np.float32)
#init_lambda_nu = tf.convert_to_tensor(_init_lambda_nu)
init_lambda_z = tf.constant(np.array(rand.dirichlet(alpha=[1.2, 0.5], size=N), dtype=np.float32))
#init_lambda_pi = tf.constant(np.array(rand.gamma(shape=5.0, scale=0.2, size=K), dtype=np.float32))
init_lambda_pi = tf.constant(np.array(rand.uniform(low=0.1, high=0.5, size=K), dtype=np.float32))
init_rho = tf.constant(0.000001)


# definition Variational parameters
with tf.name_scope("VariationalParameters"):
    #lambda_mu = tf.Variable(tf.ones([K, D]), dtype=tf.float32, trainable=True, name='lambda_mu')
    lambda_mu = tf.Variable(init_lambda_mu, dtype=tf.float32, trainable=True, name='lambda_mu')
    lambda_muLambda = tf.Variable(init_lambda_muLambda, dtype=tf.float32, trainable=True, name='lambda_muLambda')
    lambda_Lambda = tf.Variable(init_lambda_Lambda, dtype=tf.float32, trainable=True, name='lambda_Lambda')
    lambda_nu = tf.Variable(init_lambda_nu, dtype=tf.float32, trainable=True, name='lambda_nu')
    #lambda_pi = tf.Variable(tf.constant(0.5, shape=[K]), dtype=tf.float32, trainable=True, name='lambda_pi')
    lambda_pi = tf.Variable(init_lambda_pi, dtype=tf.float32, trainable=True, name='lambda_pi')
    lambda_z = tf.Variable(init_lambda_z, dtype=tf.float32, trainable=True, name='lambda_z')

# Save previous variational parameters
#prev_lambda_pi = tf.Variable(tf.ones(K), dtype=tf.float32)
#prev_lambda_mu = tf.Variable(tf.ones([D, K]), dtype=tf.float32, trainable=True)
#prev_lambda_z = tf.Variable(init_lambda_z, dtype=tf.float32)
#prev_lambda_Lambda = tf.Variable(unit_matrices, dtype=tf.float32, trainable=True)
#prev_lambda_nu = tf.Variable(tf.constant(2.0, shape=[K]), dtype=tf.float32, trainable=True)
prev_sample_q_Lambda = tf.Variable(sample_unit, dtype=tf.float32, trainable=True)

# Update counter and Learning parameter
update_counter = tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter')
rho = tf.Variable(init_rho, dtype=tf.float32, trainable=True)


# Distributions
# Approximated distributions
with tf.name_scope("ApproximatedDistributions"):
    #precision = tf.multiply(lambda_muLambda, unit_matrices, name='precision')			# precision is constraited to be diagonal matrix
    #covariance_q_mu = tf.matrix_inverse(precision, name='covariance_q_mu')
    covariance_q_mu = lambda_muLambda
    q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=lambda_mu, covariance_matrix=covariance_q_mu, name='q_mu')
    q_Lambda = tf.contrib.distributions.WishartFull(df=lambda_nu, scale=lambda_Lambda, name='q_Lambda')
    q_z = tf.contrib.distributions.OneHotCategorical(lambda_z, name='q_z')
    q_pi = tf.contrib.distributions.Dirichlet(lambda_pi, name='q_pi')

# Sampling from approximated distributions
print("* Sampling *")
with tf. name_scope('Sampling'):
    sample_q_mu = tf.Variable(tf.zeros([S, K, D]), name='sample_q_mu')
    sample_q_mu_ass = tf.assign(sample_q_mu, q_mu.sample( sample_shape=[S]), name='sample_q_mu_ass')
    sample_q_Lambda = tf.Variable(tf.ones([S, K, D, D]), name='sample_q_Lambda')
    _sample_q_Lambda = q_Lambda.sample(sample_shape=[S])
    sample_q_z = tf.Variable(tf.zeros([S, N, K]), dtype=tf.float32, name='sample_q_z')
    sample_q_z_ass = tf.assign(sample_q_z, tf.to_float(q_z.sample(sample_shape=[S])), name='sample_q_z_ass')

    sample_q_pi = tf. Variable(tf.ones([S, K]), name='sample_q_pi')
    sample_q_pi_ass = tf.assign(sample_q_pi, q_pi.sample( sample_shape=[S]), name='sample_q_pi_ass')

    # care det = 0
    det_sample_q_Lambda = tf.matrix_determinant(_sample_q_Lambda)
    det_min1 = tf.reduce_min(tf.abs(det_sample_q_Lambda))
    sample_q_Lambda_modified = tf.add(_sample_q_Lambda, det_care_diag_epsilon)
    sample_q_Lambda_new = tf.cond(tf.less(det_min1, det_lower_limit), lambda: sample_q_Lambda_modified,
                                  lambda: _sample_q_Lambda)
    sample_q_Lambda_ass = tf.assign(sample_q_Lambda, sample_q_Lambda_new)
    #det_care_c1 = tf.cond(tf.less(det_min1, det_lower_limit), lambda: tf.add(det_care_count1, 1),
    #                      lambda: det_care_count1)
    #det_care_count1_ass = tf.assign(det_care_count1, det_care_c1)

# Generative model
with tf.name_scope("GenerativeModel"):
    pred_counter = tf.to_float(tf.greater(update_counter, 0.0))[0]
    #p_Lambda = tf.contrib.distributions.WishartFull(df=hyper_nu, scale=hyper_V)
    #sample_p_Lambda = tf.Variable(tf.ones([S, K, D, D]), name='sample_p_Lambda')
    #_sample_p_Lambda = p_Lambda.sample(sample_shape=[S], seed=1)
    #sample_p_Lambda_ass = tf.assign(sample_p_Lambda, _sample_p_Lambda)
    #precision_p_mu = tf.cond(tf.equal(pred_counter, 1.0), lambda: tf.multiply(hyper_coe_alpha_var, sample_q_Lambda_ass)[0], lambda: tf.multiply(hyper_coe_alpha_var, sample_p_Lambda_ass)[0])
    #covariance_p_mu = tf.matrix_inverse(precision_p_mu)
    #p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean, covariance_matrix=covariance_p_mu)
    p_pi = tf.contrib.distributions.Dirichlet(hyper_gamma)
    pi_gene = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_pi_ass, lambda: p_pi.sample(sample_shape=[S]), name='pi_gene')  # pi for Cat(z|pi) is sampled by p_pi firstly and q_pi secondly
    #pi_gene = sample_q_pi_ass
    #mu_gene = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_mu_ass, lambda: p_mu.sample(sample_shape=[1]), name='mu_gene')  # mu for Normal(x|mu,Lambda) is sampled by p_mu firstly and q_mu secondly
    #mu_gene = sample_q_mu_ass
    #covariance_generative_gauss = tf.cond(tf.equal(pred_counter, 1.0), lambda: tf.matrix_inverse(sample_q_Lambda_ass), lambda: tf.matrix_inverse(p_Lambda.sample(sample_shape=[S])))
    #covariance_generative_gauss = tf.matrix_inverse(sample_q_Lambda_ass)
    p_z = tf.contrib.distributions.OneHotCategorical(pi_gene)  # [S,K] sample_q_pi_ass
    #generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu_gene, covariance_matrix=covariance_generative_gauss)    # [S, K, D]
    generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean,
                                                                                 covariance_matrix=tf.matrix_inverse(hyper_coe_alpha_var))  # [S, K, D]

# * Construct calculation graph for updating variational parameters *
# logarithmic distributions
print("* logarithmic distributions *")
with tf.name_scope('LogDistributions'):
    x_tile = tf.tile(X, [1, KS])
    x_tile_reshape = tf.reshape(x_tile, [N, S, K, D])   # N's [S, K, D]
    log_gene_gauss = generative_gauss.log_prob(x_tile_reshape, name='log_gene_gauss')   # [N, S, K]
    #log_gene_gauss = generative_gauss.log_prob(generative_gauss.sample(sample_shape=[N]), name='log_gene_gauss')  # [N, S, K]
    log_gene_gauss_trans = tf.transpose(log_gene_gauss, perm=[1, 0, 2]) # [S, N, K]
    sample_p_z = tf.Variable(tf.zeros([S, N, K]), dtype=tf.float32, name='sample_p_z')
    sample_p_z_ass = tf.assign(sample_p_z, tf.transpose(tf.to_float(p_z.sample(sample_shape=[N])), perm=[1, 0, 2]))
    sampled_z = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_z_ass, lambda: sample_p_z_ass)
    logpx = tf.reduce_sum(tf.multiply(tf.to_float(sampled_z), log_gene_gauss_trans), axis=2, name='logpx')
    log_p_x = tf.reduce_sum(logpx, axis=1, name='log_p_x')
    log_p_pi = p_pi.log_prob(sample_q_pi_ass, name='log_p_pi')
    sample_q_z_ass_re = tf.transpose(sampled_z, perm=[1, 0, 2])
    _log_p_z = p_z.log_prob(sample_q_z_ass_re)
    _log_p_z_trans = tf.transpose(_log_p_z, perm=[1, 0])
    log_p_z = tf.reduce_sum(_log_p_z_trans, axis=1, name='log_p_z')
    #log_p_mu = tf.reduce_sum(p_mu.log_prob(sample_q_mu_ass), axis=1, name='log_p_mu')
    #log_p_Lambda = tf.reduce_sum(p_Lambda.log_prob(sample_q_Lambda_ass), axis=1, name='log_p_Lambda')

    #log_gauss = q_mu.log_prob(sample_q_mu_ass, name='log_gauss')
    #log_q_mu = tf.reduce_sum(log_gauss, axis=1, name='log_q_mu')
    #log_wishart = q_Lambda.log_prob(sample_q_Lambda_ass, name='log_wishart')
    #log_q_Lambda = tf.reduce_sum(log_wishart, axis=1, name='log_q_Lambda')
    log_dirichlet = q_pi.log_prob(sample_q_pi_ass, name='log_dirichlet')
    log_q_pi = log_dirichlet
    log_categorical = q_z.log_prob(sample_q_z_ass, name='log_categorical')
    log_q_z = tf.reduce_sum(log_categorical, axis=1, name='log_q_z')
    #log_p = tf.add(tf.add(tf.add(tf.add(log_p_x, log_p_z), log_p_pi), log_p_mu), log_p_Lambda, name='log_p')
    #log_q = tf.add(tf.add(tf.add(log_q_z, log_q_mu), log_q_pi), log_q_Lambda, name='log_q')
    log_p = tf.add(log_p_x, tf.add(log_p_z, log_p_pi), name='log_p')
    log_q = tf.add(log_q_z, log_q_pi, name='log_q')
    log_loss = tf.subtract(log_p, log_q, name='log_loss')


# Gradients
print("* Gradients *")
with tf.name_scope('Gradients'):
    grad_q_mu = []
    grad_q_muLambda = []
    grad_q_Lambda = []
    grad_q_nu = []
    grad_q_z = []
    grad_q_pi = []
    for j in range(S):
        #grad_q_mu.append(tf.gradients(log_q_mu[j], lambda_mu, name='grad_q_mu'))
        #grad_q_muLambda.append(tf.gradients(log_q_mu[j], lambda_muLambda, name='grad_q_muLambda'))
        #grad_q_Lambda.append(tf.gradients(log_q_Lambda[j], lambda_Lambda, name='grad_q_Lambda'))
        #grad_q_nu.append(tf.gradients(log_q_Lambda[j], lambda_nu, name='gard_q_nu'))
        grad_q_z.append(tf.gradients(log_q[j], lambda_z, name='gard_q_z'))
        grad_q_pi.append(tf.gradients(log_q[j], lambda_pi, name='grad_q_pi'))


# Sample mean
print("* Sample mean *")
with tf.name_scope('SampleMean'):
    element_wise_product_mu = []
    element_wise_product_muLambda = []
    element_wise_product_Lambda = []
    element_wise_product_nu = []
    element_wise_product_z = []
    element_wise_product_pi = []
    for s in range(S):
        #element_wise_product_mu.append(tf.multiply(grad_q_mu[s][0], log_loss[s]))
        #element_wise_product_muLambda.append(tf.multiply(grad_q_muLambda[s][0], log_loss[s]))
        #element_wise_product_Lambda.append(tf.multiply(grad_q_Lambda[s][0], log_loss[s]))
        #element_wise_product_nu.append(tf.multiply(grad_q_nu[s][0], log_loss[s]))
        element_wise_product_z.append(tf.multiply(grad_q_z[s][0], log_loss[s]))
        element_wise_product_pi.append(tf.multiply(grad_q_pi[s][0], log_loss[s]))
    #sample_mean_mu = tf.reduce_mean(element_wise_product_mu, axis = 0, name='sample_mean_mu')
    #sample_mean_muLambda = tf.reduce_mean(element_wise_product_muLambda, axis=0, name='sample_mean_muLambda')
    #sample_mean_Lambda = tf.reduce_mean(element_wise_product_Lambda, axis=0, name='sample_mean_Lambda')
    #sample_mean_nu = tf.reduce_mean(element_wise_product_nu, axis=0, name='sample_mean_nu')
    sample_mean_z = tf.reduce_mean(element_wise_product_z, axis=0, name='sample_mean_z')
    sample_mean_pi = tf.reduce_mean(element_wise_product_pi, axis=0, name='sample_mean_pi')


# Update variational parameters
print("* Update variational parameters *")
with tf.name_scope('UpdateVariationalParameters'):
    #delta_mu = tf.multiply(rho, sample_mean_mu, name='delta_mu')
    #lambda_mu_update = tf.assign_add(lambda_mu, delta_mu, name='lambda_mu_update')
    #delta_muLambda = tf.multiply(rho, sample_mean_muLambda, name='delta_muLambda')
    #_lambda_muLambda_update = tf.add(lambda_muLambda, delta_muLambda, name='lambda_muLambda_update')
    #delta_Lambda = tf.multiply(rho, sample_mean_Lambda, name='delta_Lambda')
    #_lambda_Lambda_update = tf.add(lambda_Lambda, delta_Lambda, name='lambda_Lambda_update')
    #delta_nu = tf.multiply(rho, sample_mean_nu, name='delta_nu')
    #lambda_nu_update = tf.add(lambda_nu, delta_nu, name='lambda_nu_update')
    delta_z = tf.multiply(rho, sample_mean_z, name='delta_z')
    lambda_z_update = tf.add(lambda_z, delta_z, name='lambda_z_update')
    delta_pi = tf.multiply(rho, sample_mean_pi, name='delta_pi')
    lambda_pi_update = tf.add(lambda_pi, delta_pi, name='lambda_pi_update')

    ## care det = 0
    #det_lambda_muLambda_update = tf.matrix_determinant(_lambda_muLambda_update)
    #det_min2 = tf.reduce_min(tf.abs(det_lambda_muLambda_update))
    #det_care_modified2 = tf.add(_lambda_muLambda_update, det_care_diag_epsilon)
    #lambda_muLambda_update = tf.cond(tf.less(det_min2, det_lower_limit), lambda: det_care_modified2,
    #                                 lambda: _lambda_muLambda_update)
    #det_care_c2 = tf.cond(tf.less(det_min2, det_lower_limit), lambda: tf.add(det_care_count2, 1),
    #                      lambda: det_care_count2)
    #det_care_count2_ass = tf.assign(det_care_count2, det_care_c2)
    #
    #det_lambda_Lambda_update = tf.matrix_determinant(_lambda_Lambda_update)
    #det_min3 = tf.reduce_min(tf.abs(det_lambda_Lambda_update))
    #det_care_modified3 = tf.add(_lambda_Lambda_update, det_care_diag_epsilon)
    #lambda_Lambda_update = tf.cond(tf.less(det_min3, det_lower_limit), lambda: det_care_modified3,
    #                               lambda: _lambda_Lambda_update)
    #det_care_c3 = tf.cond(tf.less(det_min3, det_lower_limit), lambda: tf.add(det_care_count3, 1),
    #                      lambda: det_care_count3)
    #det_care_count3_ass = tf.assign(det_care_count3, det_care_c3)

# Update time_step and learning parameter
#update_counter_add = tf.add(update_counter, tf.ones(1))
#update_counter_ass = tf.assign(update_counter, update_counter_add)
#rho = tf.minimum(init_rho, tf.divide(tf.ones(1), tf.multiply(update_counter_ass, tf.constant(100000000.0))))
update_counter_add = tf.add(update_counter, tf.ones(1))
update_counter_ass = tf.assign(update_counter, update_counter_add)
delta_rho = tf.divide(tf.ones(1), tf.multiply(update_counter_ass, tf.constant(100000.0)))[0]
rho_ass = tf.assign(rho, tf.minimum(init_rho, delta_rho))

## Caring variational parameters
#print("* Care_lambda_muLambda *")
## * Diagonalize lambda_muLambda and transform negative eigen values to positve ones *
#with tf.name_scope('Care_lambda_muLambda'):
#    #lambda_muLambda_update_modify = tf.multiply(lambda_muLambda_update, unit_matrices)
#    _off_diagonal_lambda_muLambda = []
#    for k in range(K):
#        _off_diagonal_lambda_muLambda.append(tf.multiply(lambda_muLambda_update[k][0][1], off_diagonal[k]))
#    off_diagonal_lambda_muLambda = tf.convert_to_tensor(_off_diagonal_lambda_muLambda)
#    diag_lambda_muLambda_update = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(lambda_muLambda_update)).to_dense()
#    lambda_muLambda_update_modify = tf.add(diag_lambda_muLambda_update, off_diagonal_lambda_muLambda)
#    eig_val_lambda_muLambda = tf.self_adjoint_eig(lambda_muLambda_update_modify, name='eig_val_lambda_muLambda')[0]
#    eig_vec_lambda_muLambda = tf.self_adjoint_eig(lambda_muLambda_update_modify, name='eig_vec_lambda_muLambda')[1]
#    #_pred_eig_lambda_muLambda = []
#    #_eig_val_lambda_muLambda_modify = []
#    #for k in range(K):
#    #    _pred_eig_lambda_muLambda.append(tf.greater(eig_val_lambda_muLambda[k], tf.zeros([D])))
#    #for k in range(K):
#    #    for d in range(D):
#    #        _eig_val_lambda_muLambda_modify.append(tf.cond(
#    #            tf.equal(_pred_eig_lambda_muLambda[k][d], True),
#    #            lambda: eig_val_lambda_muLambda[k][d], lambda: epsilon))
#    eig_val_muLambda_minimum = tf.constant(0.001, shape=[K, D], dtype=tf.float32)
#    #eig_val_lambda_muLambda_modify = tf.reshape(tf.convert_to_tensor(_eig_val_lambda_muLambda_modify), shape=[K, D], name='eig_val_lambda_muLambda_modify')
#    eig_val_lambda_muLambda_modify = tf.maximum(eig_val_muLambda_minimum, eig_val_lambda_muLambda)
#    diagonalized_muLambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_muLambda_modify, name='diagonalized_muLambda').to_dense()
#    lambda_muLambda_new = tf.matmul(eig_vec_lambda_muLambda,
#                                  tf.matmul(diagonalized_muLambda, tf.matrix_inverse(eig_vec_lambda_muLambda)),
#                                  name='lambda_Lambda_new')
#    pred_lambda_muLambda = tf.to_float(tf.greater(eig_val_lambda_muLambda, tf.zeros([K, D])))
#    _lambda_muLambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_muLambda), epsilon_mKD), tf.less(tf.reduce_sum(pred_lambda_muLambda), epsilon_pKD)), lambda: lambda_muLambda_update, lambda: lambda_muLambda_new, name='mid_lambda_muLambda_cared')		# Sum of pred_lambda_muLambda is K*D if all eigenvalues are positive.
#    lambda_muLambda_cared = tf.assign(lambda_muLambda, _lambda_muLambda_cared, name='lambda_muLambda_cared')
#
#    # care det = 0
#    #det_lambda_muLambda_cared = tf.matrix_determinant(_lambda_muLambda_cared)
#    #det_min4 = tf.reduce_min(tf.abs(det_lambda_muLambda_cared))
#    #det_care_modified4 = tf.add(_lambda_muLambda_cared, det_care_diag_epsilon)
#    #det_care_new4 = tf.cond(tf.less(det_min4, det_lower_limit), lambda: det_care_modified4,
#    #                        lambda: _lambda_muLambda_cared)
#    #lambda_muLambda_cared = tf.assign(lambda_muLambda, det_care_new4, name='lambda_muLambda_cared')
#    #det_care_c4 = tf.cond(tf.less(det_min4, det_lower_limit), lambda: tf.add(det_care_count4, 1),
#    #                      lambda: det_care_count4)
#    #det_care_count4_ass = tf.assign(det_care_count4, det_care_c4)


#print("* Care lambda_Lambda *")
## * Diagonalize lambda_Lambda and transform negative values to positive ones *
#with tf.name_scope('Care_lambda_Lambda'):
#    _off_diagonal_lambda_Lambda = []
#    for k in range(K):
#        _off_diagonal_lambda_Lambda.append(tf.multiply(lambda_Lambda_update[k][0][1], off_diagonal[k]))
#    off_diagonal_lambda_Lambda = tf.convert_to_tensor(_off_diagonal_lambda_Lambda)
#    dig_lambdaLambda_update = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(lambda_Lambda_update)).to_dense()
#    lambda_Lambda_update_modify = tf.add(dig_lambdaLambda_update, off_diagonal_lambda_Lambda)
#    eig_val_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_val_lambda_Lambda')[0]
#    eig_vec_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_vec_lambda_Lambda')[1]
#    #_pred_eig_lambda_Lambda = []
#    #_eig_val_lambda_Lambda_modify = []
#    #for k in range(K):
#    #    _pred_eig_lambda_Lambda.append(tf.greater(eig_val_lambda_Lambda[k], tf.zeros([D])))
#    #for k in range(K):
#    #    for d in range(D):
#    #        _eig_val_lambda_Lambda_modify.append(tf.cond(
#    #            tf.equal(_pred_eig_lambda_Lambda[k][d], True),
#    #            lambda: eig_val_lambda_Lambda[k][d], lambda: epsilon))
#    eig_val_Lambda_minimum = tf.constant(0.001, shape=[K, D], dtype=tf.float32)
#    #eig_val_lambda_Lambda_modify = tf.reshape(tf.convert_to_tensor(_eig_val_lambda_Lambda_modify), shape=[K, D], name='eig_val_lambda_Lambda_modify')
#    eig_val_lambda_Lambda_modify = tf.maximum(eig_val_Lambda_minimum, eig_val_lambda_Lambda)
#    diagonalized_Lambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_Lambda_modify, name='diagonalized_Lambda').to_dense()
#    lambda_Lambda_new = tf.matmul(eig_vec_lambda_Lambda, tf.matmul(diagonalized_Lambda, tf.matrix_inverse(eig_vec_lambda_Lambda)), name='lambda_Lambda_new')
#    pred_lambda_Lambda = tf.to_float(tf.greater(eig_val_lambda_Lambda, tf.zeros([K, D])),
#                                 name='pred_lambda_Lambda')  # The condition of tf.cond must compare scalars but this tf.less_equal gives boolean tensor with shape [K, D].
#    _lambda_Lambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_Lambda), epsilon_mKD), tf.less(tf.reduce_sum(pred_lambda_Lambda), epsilon_pKD)), lambda: lambda_Lambda_update, lambda: lambda_Lambda_new, name='mid_lambda_Lambda_cared')		# Sum of pred_lambda_Lambda is K*D if all eigenvalues are positive.
#    _off_diagonal_lambda_Lambda_cared = []
#    for k in range(K):
#        _off_diagonal_lambda_Lambda_cared.append(tf.multiply(_lambda_Lambda_cared[k][0][1], off_diagonal[k]))
#    off_diagonal_lambda_Lambda_cared = tf.convert_to_tensor(_off_diagonal_lambda_Lambda_cared)
#    dig_lambda_Lambda_cared = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(_lambda_Lambda_cared)).to_dense()
#    _lambda_Lambda_cared_modify = tf.add(dig_lambda_Lambda_cared, off_diagonal_lambda_Lambda_cared)
#    lambda_Lambda_cared = tf.assign(lambda_Lambda, _lambda_Lambda_cared_modify, name='lambda_Lambda_cared')
#
#    # care det = 0
#    #det_lambda_Lambda_cared = tf.matrix_determinant(_lambda_Lambda_cared_modify)
#    #det_min5 = tf.reduce_min(tf.abs(det_lambda_Lambda_cared))
#    #det_care_modified5 = tf.add(_lambda_Lambda_cared_modify, det_care_diag_epsilon)
#    #det_care_new5 = tf.cond(tf.less(det_min5, det_lower_limit), lambda: det_care_modified5,
#    #                        lambda: _lambda_Lambda_cared_modify)
#    #lambda_Lambda_cared = tf.assign(lambda_Lambda, det_care_new5, name='lambda_Lambda_cared')
#    #det_care_c5 = tf.cond(tf.less(det_min5, det_lower_limit), lambda: tf.add(det_care_count5, 1),
#    #                      lambda: det_care_count5)
#    #det_care_count5_ass = tf.assign(det_care_count5, det_care_c5)
#
#    # for test output
#    det_lambda_Lambda = tf.matrix_determinant(lambda_Lambda_cared)
#    eig_val_lambda_Lambda_cared = tf.self_adjoint_eig(lambda_Lambda_cared, name='eig_val_lambda_Lambda_cared')[0]



#print("* Care lambda_nu *")
## * lambda_nu must be lager than D-1 *
#with tf.name_scope('Care_lambda_nu'):
#    dimension = tf.constant(2.0, shape=[K], dtype=tf.float32, name='dimension')
#    pred_lambda_nu = tf.to_float(tf.greater(lambda_nu_update, tf.subtract(dimension, tf.ones([K]))))
#    #pred_lambda_nu = tf.to_float(tf.greater_equal(lambda_nu_update, dimension))
#    #_lambda_nu_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_nu), epsilon_mK), tf.less(tf.reduce_sum(pred_lambda_nu), epsilon_pK)), lambda: lambda_nu_update, lambda: tf.add(tf.abs(lambda_nu_update), tf.subtract(dimension, tf.ones([K]))), name='_lambda_nu_cared')
#    #_lambda_nu_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_nu), epsilon_mK),
#    #                                          tf.less(tf.reduce_sum(pred_lambda_nu), epsilon_pK)),
#    #                           lambda: lambda_nu_update, lambda: dimension, name='mid_lambda_nu_cared')
#    _lambda_nu_cared = []
#    for k in range(K):
#        _lambda_nu_cared.append(tf.cond(tf.logical_and(tf.greater(pred_lambda_nu[k], dimension[k] - 1.0 - epsilon),
#                                                       tf.less(pred_lambda_nu[k], dimension[k] - 1.0 + epsilon)),
#                                        lambda: lambda_nu_update[k], lambda: dimension[k], name='mid_lambda_nu_cared'))
#    _lambda_nu_cared_tf = tf.convert_to_tensor(_lambda_nu_cared)
#    lambda_nu_cared = tf.assign(lambda_nu, _lambda_nu_cared_tf, name='lambda_nu_cared')


print("* Care lambda_z *")
# * lambda_z >= 0 and sum(lambda_z_{nk}, k) = 1
with tf.name_scope('Care_lambda_z'):
    _lambda_z = []
    pred_lambda_z0 = []
    _lambda_z.append(tf.split(lambda_z_update, N, 0))
    for n in range(N):
        pred_lambda_z0.append(
            tf.to_float(tf.logical_and(tf.greater(_lambda_z[0][n], tf.zeros([K])), tf.less(_lambda_z[0][n], tf.ones([K]))),
                    name='pred_lambda_z0'))
        _lambda_z[0][n] = tf.cond(tf.logical_and(tf.less(tf.reduce_sum( pred_lambda_z0[n]), epsilon_pK), tf.greater(tf.reduce_sum( pred_lambda_z0[n]), epsilon_mK)), lambda: _lambda_z[0][n], lambda: PGMethod(_lambda_z[0][n], [1, K]), name='_lambda_z')    # K_f + 0.1, K_f - 0.1
    _lambda_z_update = tf.concat(_lambda_z[0], 0, name='mid_lambda_z_update')
    lambda_z_cared = tf.assign(lambda_z, _lambda_z_update, name='lambda_z_cared')


print("* Care lambda_pi *")
# * lambda_pi >= 0 *
with tf.name_scope('Care_lambda_pi'):
    pred_lambda_pi = tf.to_float(tf.greater(lambda_pi_update, tf.zeros([K])), name='pred_lambda_pi')
    _lambda_pi_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_pi), epsilon_mK), tf.less(tf.reduce_sum(pred_lambda_pi), epsilon_pK)), lambda: lambda_pi_update, lambda: tf.abs(tf.multiply(0.5, lambda_pi_update)), name='mid_lambda_pi_cared')
    lambda_pi_cared = tf.assign(lambda_pi, _lambda_pi_cared, name='lambda_pi_cared')



# Prepare for output
print("* sample_values for output *")
with tf.name_scope('Prepare_for_output'):
    #sample_mu = q_mu.sample(sample_shape=[num_samples], name='sample_mu', seed=0)
    #sample_Sigma = tf.matrix_inverse(q_Lambda.sample( sample_shape=[num_samples], seed=0), name='sample_Sigma')
    #sample_z = q_z.sample(sample_shape=[num_samples], name='sample_z', seed=0)
    #sample_pi = q_pi.sample(sample_shape=[num_samples], name='sample_pi', seed=0)
    sample_Sigma = tf.matrix_inverse(sample_q_Lambda_ass)
    eig_Sigma = tf.self_adjoint_eigvals(sample_Sigma)
    Area_covariance_ellipse = tf.reduce_prod(eig_Sigma, axis=2)
    #eig_prev_sample_q_Lambda = tf.self_adjoint_eigvals(prev_sample_q_Lambda_ass)
    #sample_p_Lambda = p_Lambda.sample(sample_shape=[S])
    #generated_data = tf.multiply(sample_q_z, tf.transpose(generative_gauss.sample(sample_shape=[N]), perm=[3, 1, 0, 2]))
    generated_data = tf.reduce_mean(generative_gauss.sample(sample_shape=[N]), axis=1)



# * Calculation working *
print("* Session *")
# data frames for output csv
data_frame1 = pd.DataFrame(index=[], columns=['mu_class1_element1', 'mu_class1_element2', 'mu_class2_element1', 'mu_class2_element2', 'Sigma_class1_element11', 'Sigma_class1_element12', 'Sigma_class1_element21', 'Sigma_class1_element22', 'Sigma_class2_element11', 'Sigma_class2_element12', 'Sigma_class2_element21', 'Sigma_class2_element22'])		# mu and Sigma sample
data_frame2 = pd.DataFrame(index=[], columns=['z_class1', 'z_class2'])  # z sample
data_frame3 = pd.DataFrame(index=[], columns=['pi_class1', 'pi_class2'])   # pi sample
#data_frame4 = pd.DataFrame(index=[], columns=['lambda_mu_class1_element1', 'lambda_mu_class1_element2', 'lambda_mu_class2_element1', 'lambda_mu_class2_element2'])    # lambda_mu
#data_frame5 = pd.DataFrame(index=[], columns=['lambda_muLambda_class1_element11', 'lambda_muLambda_class1_element12', 'lambda_muLambda_class1_element21', 'lambda_muLambda_class1_element22', 'lambda_muLambda_class2_element11', 'lambda_muLambda_class2_element12', 'lambda_muLambda_class2_element21', 'lambda_muLambda_class2_element22'])  # lambda_muLambda
#data_frame6 = pd.DataFrame(index=[], columns=['lambda_Lambda_class1_element11', 'lambda_Lambda_class1_element12', 'lambda_Lambda_class1_element21', 'lambda_Lambda_class1_element22', 'lambda_Lambda_class2_element11', 'lambda_Lambda_class2_element12', 'lambda_Lambda_class2_element21', 'lambda_Lambda_class2_element22'])
#data_frame7 = pd.DataFrame(index=[], columns=['lambda_nu_class1', 'lambda_nu_class2'])
data_frame8 = pd.DataFrame(index=[], columns=['lambda_z_class1', 'lambda_z_class2']) # lambda_z
data_frame9 = pd.DataFrame(index=[], columns=['lambda_pi_class1', 'lambda_pi_class2'])  # lambda_pi
#data_frame10 = pd.DataFrame(index=[], columns=['eig_lambda_Lambda_class1_eig1', 'eig_lambda_Lambda_class1_eig2', 'eig_lambda_Lambda_class2_eig1', 'eig_lambda_Lambda_class2_eig2'])
data_frame11 = pd.DataFrame(index=[], columns=['log_p_mean', 'log_q_mean', 'ELBO'])
data_frame12 = pd.DataFrame(index=[], columns=["Area_covariance_ellipse_class1", "Area_covariance_ellipse_class2"])
data_frame13 = pd.DataFrame(index=[], columns=["generated_data_cl1_el1", "generated_data_cl1_el2", "generated_data_cl2_el1", "generated_data_cl2_el2"])
data_frame14 = pd.DataFrame(index=[], columns=["grad_q_z_cl1", "grad_q_z_cl2"])
data_frame15 = pd.DataFrame(index=[], columns=["grad_q_pi_cl1", "grad_q_pi_cl2"])

cnt = 0


# initialize session
sess = tf.Session()
#with tf.name_scope('summary'):
#    #summary_writer = tf.summary.FileWriter('Graph_BBVI2', tf.get_default_graph())
#    summary_op = tf.summary.merge_all()
#    log_writer = tf.summary.FileWriter('Graph_BBVI2', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.local_variables_initializer())


# session run
for epoch in range(num_epochs):
    #result = sess.run([sample_q_mu_ass, sample_Sigma, sample_q_z_ass, sample_q_pi_ass, lambda_mu_update,
    #                   lambda_muLambda_cared, lambda_Lambda_cared, lambda_nu_cared, lambda_z_cared, lambda_pi_cared,
    #                   eig_val_lambda_Lambda_cared, log_p, log_q, eig_Sigma, lambda_muLambda_update, lambda_Lambda_update,
    #                   lambda_nu_update, lambda_z_update, lambda_pi_update, rho, _lambda_Lambda_cared_modify, diagonalized_Lambda,
    #                   eig_vec_lambda_Lambda, delta_Lambda, grad_q_Lambda, log_loss, sample_mean_Lambda, log_p_Lambda, det_lambda_Lambda, log_p_x,
    #                   det_care_count1_ass, det_care_count2_ass, det_care_count3_ass, Area_covariance_ellipse, sample_p_Lambda], feed_dict={X: x})
    result = sess.run([sample_q_mu_ass, sample_Sigma, sample_q_z_ass, sample_q_pi_ass, lambda_z_cared, lambda_pi_cared,
                       log_p, log_q, lambda_z_update, lambda_pi_update, rho_ass, update_counter_ass,
                       grad_q_pi, grad_q_z, sample_mean_pi, sample_mean_z, delta_pi, delta_z, log_loss, log_p_x,
                       logpx, Area_covariance_ellipse, generated_data], feed_dict={X: x})
    for sam in range(num_samples):
        series1 = pd.Series([result[0][sam][0][0], result[0][sam][0][1], result[0][sam][1][0], result[0][sam][1][1],
                             result[1][sam][0][0][0], result[1][sam][0][0][1], result[1][sam][0][1][0], result[1][sam][0][1][1],
                             result[1][sam][1][0][0], result[1][sam][1][0][1], result[1][sam][1][1][0],
                             result[1][sam][1][1][1]], index=data_frame1.columns)
        data_frame1 = data_frame1.append(series1, ignore_index=True)
    
    if epoch == num_epochs-1:
        for n in range(N):
            series2 = pd.Series([1.0*(np.mean(result[2], axis=0)[n][0] >= 0.5), 1.0*(np.mean(result[2], axis=0)[n][1] > 0.5)],
                            index=data_frame2.columns)
            data_frame2 = data_frame2.append(series2, ignore_index=True)

    for sam in range(num_samples):
        series3 = pd.Series([result[3][sam][0], result[3][sam][1]], index=data_frame3.columns)
        data_frame3 = data_frame3.append(series3, ignore_index=True)

    #series4 = pd.Series(
    #    [result[4][0][0], result[4][0][1], result[4][1][0], result[4][1][1]],
    #    index=data_frame4.columns)
    #data_frame4 = data_frame4.append(series4, ignore_index=True)
    #series5 = pd.Series(
    #    [result[5][0][0][0], result[5][0][0][1], result[5][0][1][0], result[5][0][1][1], result[5][1][0][0],
    #     result[5][1][0][1], result[5][1][1][0], result[5][1][1][1]], index=data_frame5.columns)
    #data_frame5 = data_frame5.append(series5, ignore_index=True)
    #series6 = pd.Series(
    #    [result[6][0][0][0], result[6][0][0][1], result[6][0][1][0], result[6][0][1][1], result[6][1][0][0],
    #     result[6][1][0][1], result[6][1][1][0], result[6][1][1][1]], index=data_frame6.columns)
    #data_frame6 = data_frame6.append(series6, ignore_index=True)
    #series7 = pd.Series([result[7][0], result[7][1]], index=data_frame7.columns)
    #data_frame7 = data_frame7.append(series7, ignore_index=True)

    if epoch == num_epochs-1:
        for m in range(N):
            series8 = pd.Series([result[4][m][0], result[4][m][1]], index=data_frame8.columns)
            data_frame8 = data_frame8.append(series8, ignore_index=True)
    
    series9 = pd.Series([result[5][0], result[5][1]], index=data_frame9.columns)
    data_frame9 = data_frame9.append(series9, ignore_index=True)
    #series10 = pd.Series(
    #    [result[10][0][0], result[10][0][1], result[10][1][0], result[10][1][1]],
    #    index=data_frame10.columns)
    #data_frame10 = data_frame10.append(series10, ignore_index=True)
    elbo = cal_elbo.ELBO(result[6], result[7])
    series11 = pd.Series([np.mean(result[6]), np.mean(result[7]), elbo], index=data_frame11.columns)
    data_frame11 = data_frame11.append(series11, ignore_index=True)

    series12 = pd.Series([np.mean(result[21], axis=0)[0], np.mean(result[21], axis=0)[1]], index=data_frame12.columns)
    data_frame12 = data_frame12.append(series12, ignore_index=True)
    
    if epoch == num_epochs-1:
        for n in range(N):
            series13 = pd.Series([result[22][n][0][0], result[22][n][0][1], result[22][n][1][0], result[22][n][1][1]], index=data_frame13.columns)
            data_frame13 = data_frame13.append(series13, ignore_index=True)

    for n in range(N):
        if n % 100 == 0:
            series14 = pd.Series([np.mean(result[13], axis=0)[0][n][0], np.mean(result[13], axis=0)[0][n][1]], index=data_frame14.columns)
            data_frame14 = data_frame14.append(series14, ignore_index=True)

    series15 = pd.Series([np.mean(result[12], axis=0)[0][0], np.mean(result[12], axis=0)[0][1]], index=data_frame15.columns)
    data_frame15 = data_frame15.append(series15, ignore_index=True)


    #print(result)
    print("sample_q_mu")
    print(result[0])
    print("sample_Sigma")
    print(result[1])
    print("sample_q_z")
    print(result[2])
    print("sample_q_pi")
    print(result[3])
    #print("lambda_mu_update")
    #print(result[4])
    #print("lambda_muLambda_cared")
    #print(result[5])
    #print("lambda_Lambda_cared")
    #print(result[6])
    #print("lambda_nu_cared")
    #print(result[7])
    print("lambda_z_cared")
    print(result[4])
    print("lambda_pi_cared")
    print(result[5])
    #print("eig_val_lambda_Lambda_cared")
    #print(result[10])
    print("log_p")
    print(result[6])
    print("log_q")
    print(result[7])
    #print("lambda_muLambda_update")
    #print(result[14])
    #print("lambda_Lambda_update")
    #print(result[15])
    #print("lambda_nu_update")
    #print(result[16])
    print("lambda_z_update")
    print(result[8])
    print("lambda_pi_update")
    print(result[9])
    print("rho_ass")
    print(result[10])
    print("update_counter_ass")
    print(result[11])
    print("grad_q_pi")
    print(result[12])
    print("grad_q_z")
    print(result[13])
    print("sample_mean_pi")
    print(result[14])
    print("sample_mean_z")
    print(result[15])
    print("delta_pi")
    print(result[16])
    print("delta_z")
    print(result[17])
    print("log_loss")
    print(result[18])
    print("log_p_x")
    print(result[19])
    print("logpx")
    print(result[20])
    print("Area_covariance_ellipse")
    print(result[21])
    print("generated_data")
    print(result[22])
    

    cnt = cnt + 1
    print("epoch %d" % cnt)
    # log_writer = tf.summary.FileWriter('Graph_BBVI2', sess.graph)
    # log_writer.add_summary(summary, epoch)

# write csv
data_frame1.to_csv("res_mu_and_Sigma_K2.csv", index=False)
data_frame2.to_csv("res_z_K2.csv", index=False)
data_frame3.to_csv("res_pi_K2.csv", index=False)
#data_frame4.to_csv("res_lambda_mu_K2.csv", index=False)
#data_frame5.to_csv("res_lambda_muLambda_K2.csv", index=False)
#data_frame6.to_csv("res_lambda_Lambda_K2.csv", index=False)
#data_frame7.to_csv("res_lambda_nu_K2.csv", index=False)
data_frame8.to_csv("res_lambda_z_K2.csv", index=False)
data_frame9.to_csv("res_lambda_pi_K2.csv", index=False)
#data_frame10.to_csv("res_lambda_Lambda_eigval_K2.csv", index=False)
data_frame11.to_csv("res_ELBO_K2.csv", index=False)
data_frame12.to_csv("res_Area_covariance_ellipse.csv", index=False)
data_frame13.to_csv("res_generated_data.csv", index=False)
data_frame14.to_csv("res_grad_q_z.csv", index=False)
data_frame15.to_csv("res_grad_q_pi.csv", index=False)


