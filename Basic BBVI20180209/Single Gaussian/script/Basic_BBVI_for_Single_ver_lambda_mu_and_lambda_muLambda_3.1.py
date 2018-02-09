import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.random as rand
import scipy.stats as stats
import Input_data_for_2_dim_Gauss_test as inp
import calc_ELBO as cal_elbo


# * Initialize *
print("* Initialize *")
# Constants
N = 1000  	# number of data points
D = 2     	# dimensionality of data
S = 100		# sample
_alpha = 0.0
_beta = tf.constant(10.0, shape=[D, D])
num_epochs = 1000
num_samples = S
epsilon = tf.constant(0.0001)    # 10^{-4}
D_f = tf.to_float(D)
epsilon_plus = tf.add(1.0, epsilon)
epsilon_minus = tf.subtract(1.0, epsilon)
epsilon_pD = tf.add(D_f, epsilon)
epsilon_mD = tf.subtract(D_f, epsilon)


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
unit_matrices = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([D, D])), name='unit_matrices').to_dense()
off_diagonal = tf.subtract(tf.ones([D, D]), unit_matrices)
sample_unit = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(tf.ones([S, D, D])), name='sample_unit').to_dense()
hyper_alpha_mean = tf.constant([3.1, 3.0], shape=[D], dtype=tf.float32, name='hyper_alpha_mean')
hyper_coe_alpha_var = tf.multiply(unit_matrices, _beta, name='hyper_coe_alpha_var')
hyper_V = tf.constant([[0.15, 0.0], [0.0, 0.2]])
hyper_nu = tf.constant(3.0)
init_lambda_mu = tf.convert_to_tensor(np.array(rand.multivariate_normal(mean=[1.0, 1.0], cov=[[0.1, 0.0], [0.0, 0.1]]), dtype=np.float32))
#init_lambda_mu = tf.constant([1.0, 1.0])
#init_lambda_muLambda = tf.constant([[3.0, 0.0], [0.0, 3.0]])
#disp_init_lambda_muLambda = 5.0*tf.matrix_inverse(tf.convert_to_tensor(np.array(stats.wishart.rvs(df=20.0, scale=[[0.1, 0.0], [0.0, 0.1]]), dtype=np.float32)))
#init_lambda_muLambda = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(disp_init_lambda_muLambda)).to_dense()
init_lambda_muLambda = tf.multiply(tf.convert_to_tensor(np.array(rand.uniform(low=3.0, high=5.0, size=1), dtype=np.float32)), unit_matrices)
#init_lambda_Lambda = tf.convert_to_tensor(np.array(stats.wishart.rvs(df=20.0, scale=[[0.5, 0.0], [0.0, 0.5]]), dtype=np.float32))
init_lambda_Lambda = tf.constant([[0.2, 0.0], [0.0, 0.2]], dtype=tf.float32)
#_init_lambda_nu = np.array(rand.uniform(low=D-1+0.001, high=30.0, size=1), dtype=np.float32)
#init_lambda_nu = tf.convert_to_tensor(_init_lambda_nu)[0]
init_lambda_nu = tf.constant(10.0)
init_rho = tf.constant(0.00001)


# definition Variational parameters
with tf.name_scope("VariationalParameters"):
    lambda_mu = tf.Variable(init_lambda_mu, dtype=tf.float32, trainable=True, name='lambda_mu')
    lambda_muLambda = tf.Variable(init_lambda_muLambda, dtype=tf.float32, trainable=True, name='lambda_muLambda')
    lambda_Lambda = tf.Variable(init_lambda_Lambda, dtype=tf.float32, trainable=True, name='lambda_Lambda')
    lambda_nu = tf.Variable(init_lambda_nu, dtype=tf.float32, trainable=True, name='lambda_nu')
    
    
# Save previous variational parameters
prev_sample_q_Lambda = tf.Variable(sample_unit, dtype=tf.float32, trainable=True)

# Update counter and Learning parameter
update_counter = tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=True, name='update_counter')
rho = tf.Variable(init_rho, dtype=tf.float32, trainable=True)


# Distributions
# Approximated distributions
with tf.name_scope("ApproximatedDistributions"):
    #precision = tf.multiply(lambda_muLambda, unit_matrices, name='precision')			# precision is constraited to be diagonal matrix
    #covariance_q_mu = tf.matrix_inverse(precision, name='covariance_q_mu')
    #covariance_q_mu = tf.multiply(lambda_muLambda, unit_matrices, name='covariance_q_mu')
    covariance_q_mu = lambda_muLambda
    q_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=lambda_mu, covariance_matrix=covariance_q_mu, name='q_mu')
    q_Lambda = tf.contrib.distributions.WishartFull(df=lambda_nu, scale=lambda_Lambda, name='q_Lambda')

# Sampling from approximated distributions
print("* Sampling *")
with tf. name_scope('Sampling'):
    sample_q_mu = tf.Variable(tf.zeros([S, D]), name='sample_q_mu')
    sample_q_mu_ass = tf.assign(sample_q_mu, q_mu.sample(sample_shape=[S], seed=0), name='sample_q_mu_ass')
    sample_q_Lambda = tf.Variable(tf.ones([S, D, D]), name='sample_q_Lambda')
    _sample_q_Lambda = q_Lambda.sample(sample_shape=[S], seed=0)
    pred_q_Lambda_sample = tf.to_int32(tf.greater(tf.matrix_determinant(_sample_q_Lambda), tf.constant(0.00001, shape=[S])))
    sample_q_Lambda_cond = tf.cond(tf.equal(tf.reduce_sum(pred_q_Lambda_sample), S),
                                   lambda: _sample_q_Lambda, lambda: prev_sample_q_Lambda)
    sample_q_Lambda_ass = tf.assign(sample_q_Lambda, sample_q_Lambda_cond,
                                    name='sample_q_Lambda_ass')
    prev_sample_q_Lambda_ass = tf.assign(prev_sample_q_Lambda, sample_q_Lambda_ass, name='prev_sample_q_Lambda')

# Generative model
with tf.name_scope("GenerativeModel"):
    pred_counter = tf.to_float(tf.greater(update_counter, 0.0))[0]
    #p_Lambda = tf.contrib.distributions.WishartFull(df=hyper_nu, scale=hyper_V)
    #sample_p_Lambda = tf.Variable(tf.ones([S, D, D]), name='sample_p_Lambda')
    #_sample_p_Lambda = p_Lambda.sample(sample_shape=[S], seed=3)
    #sample_p_Lambda_ass = tf.assign(sample_p_Lambda, _sample_p_Lambda)
    #precision_p_mu = tf.cond(tf.equal(pred_counter, 1.0), lambda: tf.multiply(hyper_coe_alpha_var, sample_q_Lambda_ass)[0], lambda: tf.multiply(hyper_coe_alpha_var, sample_p_Lambda_ass)[0])
    #covariance_p_mu = tf.matrix_inverse(precision_p_mu)
    covariance_p_mu = tf.matrix_inverse(hyper_coe_alpha_var)
    #p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean, covariance_matrix=covariance_p_mu)
    p_mu = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=hyper_alpha_mean,
                                                                     covariance_matrix=covariance_p_mu)
    mu_gene = tf.cond(tf.equal(pred_counter, 1.0), lambda: sample_q_mu_ass, lambda: p_mu.sample(sample_shape=[S]), name='mu_gene')  # mu for Normal(x|mu,Lambda) is sampled by p_mu firstly and q_mu secondly
    #mu_gene = sample_q_mu_ass
    #inv_sample_q_Lambda_ass = tf.matrix_inverse(sample_q_Lambda_ass)
    #det_sample_p_Lambda = tf.matrix_determinant(sample_p_Lambda_ass)
    #inv_sample_p_Lambda = tf.matrix_inverse(sample_p_Lambda_ass)
    #cond_counter = tf.equal(pred_counter, 1.0)
    #covariance_generative_gauss = tf.cond(cond_counter, lambda: inv_sample_q_Lambda_ass, lambda: inv_sample_p_Lambda)
    #covariance_generative_gauss = tf.matrix_inverse(sample_q_Lambda_ass)
    #generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu_gene, covariance_matrix=covariance_generative_gauss)    # [S, N, D]
    generative_gauss = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu_gene,
                                                                                 covariance_matrix=tf.matrix_inverse(hyper_coe_alpha_var))  # [S, N, D]

# * Construct calculation graph for updating variational parameters *
# logarithmic distributions
print("* logarithmic distributions *")
with tf.name_scope('LogDistributions'):
    x_tile = tf.tile(X, [1, S])
    x_tile_reshape = tf.reshape(x_tile, [N, S, D])   # N's [S, D]
    log_gene_gauss = generative_gauss.log_prob(x_tile_reshape, name='log_gene_gauss')   # [N, S, D]
    log_gene_gauss_trans = tf.transpose(log_gene_gauss, perm=[1, 0]) # [S, N, D]
    log_p_x = tf.reduce_sum(log_gene_gauss_trans, axis=1, name='log_p_x')
    log_p_mu = p_mu.log_prob(sample_q_mu_ass)
    #log_p_Lambda = p_Lambda.log_prob(sample_q_Lambda_ass)

    log_gauss = q_mu.log_prob(sample_q_mu_ass, name='log_gauss')
    log_q_mu = log_gauss
    #log_wishart = q_Lambda.log_prob(sample_q_Lambda_ass, name='log_wishart')
    #log_q_Lambda = log_wishart
    #log_p = tf.add(tf.add(log_p_x, log_p_mu), log_p_Lambda, name='log_p')
    log_p = tf.add(log_p_x, log_p_mu, name='log_p')
    #log_q = tf.add(log_q_mu, log_q_Lambda, name='log_q')
    log_q = log_q_mu
    log_loss = tf.subtract(log_p, log_q, name='log_loss')


# Gradients
print("* Gradients *")
with tf.name_scope('Gradients'):
    grad_q_mu = []
    grad_q_muLambda = []
    grad_q_Lambda = []
    grad_q_nu = []
    for j in range(S):
        grad_q_mu.append(tf.gradients(log_q[j], lambda_mu, name='grad_q_mu'))
        grad_q_muLambda.append(tf.gradients(log_q[j], lambda_muLambda, name='grad_q_muLambda'))
        #grad_q_Lambda.append(tf.gradients(log_q[j], lambda_Lambda, name='grad_q_Lambda'))
        #grad_q_nu.append(tf.gradients(log_q[j], lambda_nu, name='gard_q_nu'))


# Sample mean
print("* Sample mean *")
with tf.name_scope('SampleMean'):
    element_wise_product_mu = []
    element_wise_product_muLambda = []
    element_wise_product_Lambda = []
    element_wise_product_nu = []
    for s in range(S):
        element_wise_product_mu.append(tf.multiply(grad_q_mu[s][0], log_loss[s]))				# Why can use tf.multiply between different type tensors?
        element_wise_product_muLambda.append(tf.multiply(grad_q_muLambda[s][0], log_loss[s]))
        #element_wise_product_Lambda.append(tf.multiply(grad_q_Lambda[s][0], log_loss[s]))
        #element_wise_product_nu.append(tf.multiply(grad_q_nu[s][0], log_loss[s]))
    sample_mean_mu = tf.reduce_mean(element_wise_product_mu, axis = 0, name='sample_mean_mu')
    sample_mean_muLambda = tf.reduce_mean(element_wise_product_muLambda, axis=0, name='sample_mean_muLambda')
    #sample_mean_Lambda = tf.reduce_mean(element_wise_product_Lambda, axis=0, name='sample_mean_Lambda')
    #sample_mean_nu = tf.reduce_mean(element_wise_product_nu, axis=0, name='sample_mean_nu')


# Update variational parameters
print("* Update variational parameters *")
with tf.name_scope('UpdateVariationalParameters'):
    delta_mu = tf.multiply(rho, sample_mean_mu, name='delta_mu')
    lambda_mu_update = tf.assign_add(lambda_mu, delta_mu, name='lambda_mu_update')
    delta_muLambda = tf.multiply(rho, sample_mean_muLambda, name='delta_muLambda')
    lambda_muLambda_update = tf.add(lambda_muLambda, delta_muLambda, name='lambda_muLambda_update')
    #delta_Lambda = tf.multiply(rho, sample_mean_Lambda, name='delta_Lambda')
    #lambda_Lambda_update = tf.add(lambda_Lambda, delta_Lambda, name='lambda_Lambda_update')
    #delta_nu = tf.multiply(rho, sample_mean_nu, name='delta_nu')
    #lambda_nu_update = tf.add(lambda_nu, delta_nu, name='lambda_nu_update')


# Update time_step and learning parameter
update_counter_add = tf.add(update_counter, tf.ones(1))
update_counter_ass = tf.assign(update_counter, update_counter_add)
delta_rho = tf.divide(tf.ones(1), tf.multiply(update_counter_ass, tf.constant(10000.0)))[0]
rho_ass = tf.assign(rho, tf.minimum(init_rho, delta_rho))


# Caring variational parameters
print("* Care_lambda_muLambda *")
# * Diagonalize lambda_muLambda and transform negative eigen values to positve ones *
with tf.name_scope('Care_lambda_muLambda'):
    #_lambda_muLambda_update = tf.multiply(lambda_muLambda_update, unit_matrices)
    eig_val_lambda_muLambda = tf.self_adjoint_eigvals(lambda_muLambda_update, name='eig_val_lambda_muLambda')		# eigenvalues of lambda_muLambda
    eig_val_muLambda_minimum = tf.constant(0.000001, shape=[D], dtype=tf.float32)
    eig_val_lambda_muLambda_modify = tf.maximum(eig_val_muLambda_minimum, eig_val_lambda_muLambda)
    diagonalized_muLambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_muLambda_modify, name='diagonalized_muLambda').to_dense()
    pred_lambda_muLambda = tf.to_float(tf.greater(eig_val_lambda_muLambda, tf.zeros([D])))
    #_lambda_muLambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_muLambda), epsilon_mD), tf.less(tf.reduce_sum(pred_lambda_muLambda), epsilon_pD)), lambda: lambda_muLambda_update, lambda: diagonalized_muLambda, name='_lambda_muLambda_cared')		# Sum of pred_lambda_muLambda is K*D if all eigenvalues are positive.
    _lambda_muLambda_cared = diagonalized_muLambda
    lambda_muLambda_cared = tf.assign(lambda_muLambda, _lambda_muLambda_cared, name='lambda_muLambda_cared')
    


print("* Care lambda_Lambda *")
# * Diagonalize lambda_Lambda and transform negative values to positive ones *
#with tf.name_scope('Care_lambda_Lambda'):
    #_off_diagonal_lambda_Lambda = []
    #_off_diagonal_lambda_Lambda.append(tf.multiply(lambda_Lambda_update[0][1], off_diagonal))
    #off_diagonal_lambda_Lambda = tf.convert_to_tensor(_off_diagonal_lambda_Lambda)
    #dig_lambdaLambda_update = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(lambda_Lambda_update)).to_dense()
    #lambda_Lambda_update_modify = tf.add(dig_lambdaLambda_update, off_diagonal_lambda_Lambda)
    #eig_val_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_val_lambda_Lambda')[0]		# eigen values of lambda_Lambda
    #eig_vec_lambda_Lambda = tf.self_adjoint_eig(lambda_Lambda_update_modify, name='eig_vec_lambda_Lambda')[1]		# eigen vectors of lambda_Lambda and this is orthonormal matrix which diagonalize lambda_Lambda
    ##_pred_eig_lambda_Lambda = []
    ##_eig_val_lambda_Lambda_modify = []
    ##for k in range(K):
    ##    _pred_eig_lambda_Lambda.append(tf.greater(eig_val_lambda_Lambda[k], tf.zeros([D])))
    ##for k in range(K):
    ##    for d in range(D):
    ##        _eig_val_lambda_Lambda_modify.append(tf.cond(
    ##            tf.equal(_pred_eig_lambda_Lambda[k][d], True),
    ##            lambda: eig_val_lambda_Lambda[k][d], lambda: epsilon))
    #eig_val_Lambda_minimum = tf.constant(0.000001, shape=[D], dtype=tf.float32)
    #eig_val_lambda_Lambda_modify = tf.maximum(eig_val_Lambda_minimum, eig_val_lambda_Lambda)
    #diagonalized_Lambda = tf.contrib.linalg.LinearOperatorDiag(eig_val_lambda_Lambda_modify, name='diagonalized_Lambda').to_dense()
    #lambda_Lambda_new = tf.matmul(eig_vec_lambda_Lambda, tf.matmul(diagonalized_Lambda, tf.matrix_inverse(eig_vec_lambda_Lambda)), name='lambda_Lambda_new')
    #pred_lambda_Lambda = tf.to_float(tf.greater(eig_val_lambda_Lambda, tf.zeros([D])),
    #                             name='pred_lambda_Lambda')  # The condition of tf.cond must compare scalars but this tf.less_equal gives boolean tensor with shape [D].
    #_lambda_Lambda_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_Lambda), epsilon_mD), tf.less(tf.reduce_sum(pred_lambda_Lambda), epsilon_pD)), lambda: lambda_Lambda_update, lambda: lambda_Lambda_new, name='_lambda_Lambda_cared')		# Sum of pred_lambda_Lambda is D if all eigenvalues are positive.
    #_off_diagonal_lambda_Lambda_cared = []
    #_off_diagonal_lambda_Lambda_cared.append(tf.multiply(_lambda_Lambda_cared[0][1], off_diagonal))
    #off_diagonal_lambda_Lambda_cared = tf.convert_to_tensor(_off_diagonal_lambda_Lambda_cared)
    #dig_lambda_Lambda_cared = tf.contrib.linalg.LinearOperatorDiag(tf.matrix_diag_part(_lambda_Lambda_cared)).to_dense()
    #_lambda_Lambda_cared_modify = tf.add(dig_lambda_Lambda_cared, off_diagonal_lambda_Lambda_cared)
    #lambda_Lambda_cared = tf.assign(lambda_Lambda, _lambda_Lambda_cared_modify[0], name='lambda_Lambda_cared')
    #det_lambda_Lambda = tf.matrix_determinant(lambda_Lambda_cared)
    #eig_val_lambda_Lambda_cared = tf.self_adjoint_eig(lambda_Lambda_cared, name='eig_val_lambda_Lambda_cared')[0]
    


print("* Care lambda_nu *")
# * lambda_nu must be lager than D-1 *
#with tf.name_scope('Care_lambda_nu'):
#    dimension = tf.constant(2.0, dtype=tf.float32, name='dimension')
#    pred_lambda_nu = tf.to_float(tf.greater(lambda_nu_update, tf.subtract(dimension, tf.ones([1]))))
#    _lambda_nu_cared = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(pred_lambda_nu), epsilon_minus),
#                                              tf.less(tf.reduce_sum(pred_lambda_nu), epsilon_plus)),
#                               lambda: lambda_nu_update, lambda: dimension, name='_lambda_nu_cared')
#    lambda_nu_cared = tf.assign(lambda_nu, _lambda_nu_cared, name='lambda_nu_cared')



# Prepare for output
print("* sample_values for output *")
with tf.name_scope('Prepare_for_output'):
    sample_Sigma = tf.matrix_inverse(sample_q_Lambda_ass)
    eig_Sigma = tf.self_adjoint_eigvals(sample_Sigma)
    Area_covariance_ellipse = tf.multiply(tf.constant(np.pi), tf.reduce_prod(tf.sqrt(eig_Sigma), axis=1))



# * Calculation working *
print("* Session *")
# data frames for output csv
data_frame1 = pd.DataFrame(index=[], columns=['mu_element1', 'mu_element2', 'Sigma_element11', 'Sigma_element12', 'Sigma_element21', 'Sigma_element22'])		# mu and Sigma sample
data_frame2 = pd.DataFrame(index=[], columns=['lambda_mu_element1', 'lambda_mu_element2'])    # lambda_mu
data_frame3 = pd.DataFrame(index=[], columns=['lambda_muLambda_element11', 'lambda_muLambda_element12', 'lambda_muLambda_element21', 'lambda_muLambda_element22'])  # lambda_muLambda
#data_frame4 = pd.DataFrame(index=[], columns=['lambda_Lambda_element11', 'lambda_Lambda_element12', 'lambda_Lambda_element21', 'lambda_Lambda_element22'])
#data_frame5 = pd.DataFrame(index=[], columns=['lambda_nu'])
data_frame6 = pd.DataFrame(index=[], columns=['log_p_mean', 'log_q_mean', 'ELBO'])
data_frame7 = pd.DataFrame(index=[], columns=['Area_ellipse_covariance'])
data_frame8 = pd.DataFrame(index=[], columns=['grad_q_mu_element1', 'grad_q_mu_element2'])
data_frame9 = pd.DataFrame(index=[], columns=['grad_q_muLambda_el11', 'grad_q_muLambda_el12', 'grad_q_muLambda_el21', 'grad_q_muLambda_el22'])
data_frame10 = pd.DataFrame(index=[], columns=['sample_mean_mu_el1', 'sample_mean_mu_el2'])
data_frame11 = pd.DataFrame(index=[], columns=['sample_mean_muLambda_el11', 'sample_mean_muLambda_el12', 'sample_mean_muLambda_el21', 'sample_mean_muLambda_el22'])
data_frame12 = pd.DataFrame(index=[], columns=['delta_mu_el1', 'delta_mu_el2'])
data_frame13 = pd.DataFrame(index=[], columns=['delta_muLambda_el11', 'delta_muLambda_el12', 'delta_muLambda_el21', 'delta_muLambda_el22'])

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
    #result = sess.run([sample_q_mu_ass, sample_Sigma, lambda_mu_update,
    #                   lambda_muLambda_cared, lambda_Lambda_cared, lambda_nu_cared,
    #                   eig_val_lambda_Lambda_cared, log_p, log_q, eig_Sigma, lambda_muLambda_update, lambda_Lambda_update,
    #                   lambda_nu_update, log_p_x, log_p_Lambda, Area_covariance_ellipse], feed_dict={X: x})		# for plot test
    result = sess.run([sample_q_mu_ass, sample_Sigma, lambda_mu_update, lambda_muLambda_cared,
                       log_p, log_q, eig_Sigma, lambda_muLambda_update,
                       log_p_x, Area_covariance_ellipse,
                       update_counter_ass, rho_ass, grad_q_mu, grad_q_muLambda, sample_mean_mu, sample_mean_muLambda,
                       delta_mu, delta_muLambda],
                      feed_dict={X: x})  # for plot test

    for sam in range(num_samples):
        series1 = pd.Series([result[0][sam][0], result[0][sam][1], result[1][sam][0][0],
                             result[1][sam][0][1], result[1][sam][1][0], result[1][sam][1][1],
                             ], index=data_frame1.columns)
        data_frame1 = data_frame1.append(series1, ignore_index=True)


    series2 = pd.Series(
        [result[2][0], result[2][1]],
        index=data_frame2.columns)
    data_frame2 = data_frame2.append(series2, ignore_index=True)
    series3 = pd.Series(
        [result[3][0][0], result[3][0][1], result[3][1][0], result[3][1][1]], index=data_frame3.columns)
    data_frame3 = data_frame3.append(series3, ignore_index=True)
    #series4 = pd.Series(
    #    [result[4][0][0], result[4][0][1], result[4][1][0], result[4][1][1]], index=data_frame4.columns)
    #data_frame4 = data_frame4.append(series4, ignore_index=True)
    #series5 = pd.Series(result[5], index=data_frame5.columns)
    #data_frame5 = data_frame5.append(series5, ignore_index=True)

    #elbo = cal_elbo.ELBO(result[7], result[8])
    #series6 = pd.Series([np.mean(result[7]), np.mean(result[8]), elbo], index=data_frame6.columns)
    #data_frame6 = data_frame6.append(series6, ignore_index=True)
    elbo = cal_elbo.ELBO(result[4], result[5])
    series6 = pd.Series([np.mean(result[4]), np.mean(result[5]), elbo], index=data_frame6.columns)
    data_frame6 = data_frame6.append(series6, ignore_index=True)

    #for s in range(num_samples):
    #    series7 = pd.Series(np.mean(result[15], axis=0), index=data_frame7.columns)
    #    data_frame7 = data_frame7.append(series7, ignore_index=True)
    for s in range(num_samples):
        series7 = pd.Series(np.mean(result[9], axis=0), index=data_frame7.columns)
        data_frame7 = data_frame7.append(series7, ignore_index=True)

    for s in range(num_samples):
        series8 = pd.Series([result[12][s][0][0], result[12][s][0][1]], index=data_frame8.columns)
        data_frame8 = data_frame8.append(series8, ignore_index=True)

    for s in range(num_samples):
        series9 = pd.Series([result[13][s][0][0][0], result[13][s][0][0][1], result[13][s][0][1][0], result[13][s][0][1][1]], index=data_frame9.columns)
        data_frame9 = data_frame9.append(series9, ignore_index=True)

    series10 = pd.Series([result[14][0], result[14][1]], index=data_frame10.columns)
    data_frame10 = data_frame10.append(series10, ignore_index=True)

    series11 = pd.Series([result[15][0][0], result[15][0][1], result[15][1][0], result[15][1][1]], index=data_frame11.columns)
    data_frame11 = data_frame11.append(series11, ignore_index=True)

    series12 = pd.Series([result[16][0], result[16][1]], index=data_frame12.columns)
    data_frame12 = data_frame12.append(series12, ignore_index=True)

    series13 = pd.Series([result[17][0][0], result[17][0][1], result[17][1][0], result[17][1][1]], index=data_frame13.columns)
    data_frame13 = data_frame13.append(series13, ignore_index=True)


    #print(result)
    print("sample_q_mu")
    print(result[0])
    print("sample_Sigma")
    print(result[1])
    print("lambda_mu_update")
    print(result[2])
    print("lambda_muLambda_cared")
    print(result[3])
    #print("lambda_Lambda_cared")
    #print(result[4])
    #print("lambda_nu_cared")
    #print(result[5])
    #print("eig_val_lambda_Lambda_cared")
    #print(result[5])
    print("log_p")
    print(result[4])
    print("log_q")
    print(result[5])
    print("eig_Sigma")
    print(result[6])
    print("lambda_muLambda_update")
    print(result[7])
    #print("lambda_Lambda_update")
    #print(result[10])
    print("log_p_x")
    print(result[8])
    #print("log_p_Lambda")
    #print(result[12])
    print("Area_covariance_ellipse")
    print(result[9])
    print("update_counter_ass")
    print(result[10])
    print("rho_ass")
    print(result[11])




    cnt = cnt + 1
    print("epoch %d" % cnt)
    # log_writer = tf.summary.FileWriter('Graph_BBVI2', sess.graph)
    # log_writer.add_summary(summary, epoch)

# write csv
data_frame1.to_csv("res_mu_and_Sigma_single_Gauss.csv", index=False)
data_frame2.to_csv("res_lambda_mu_single_Gauss.csv", index=False)
data_frame3.to_csv("res_lambda_muLambda_single_Gauss.csv", index=False)
#data_frame4.to_csv("res_lambda_Lambda_single_Gauss.csv", index=False)
#data_frame5.to_csv("res_lambda_nu_single_Gauss.csv", index=False)
data_frame6.to_csv("res_ELBO_single_Gauss.csv", index=False)
data_frame7.to_csv("res_Area_covariance_ellipse_single_Gauss.csv", index=False)
data_frame8.to_csv("res_grad_q_mu_single.csv", index=False)
data_frame9.to_csv("res_grad_q_muLambda_single.csv", index=False)
data_frame10.to_csv("res_sample_mean_mu_single.csv", index=False)
data_frame11.to_csv("res_sample_mean_muLambda_single.csv", index=False)
data_frame12.to_csv("res_delta_mu_single.csv", index=False)
data_frame13.to_csv("res_delta_muLambda_single.csv", index=False)

