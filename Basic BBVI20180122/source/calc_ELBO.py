# Evidence Lower Bound
import numpy as np


def ELBO(log_p, log_q):
    expectation_log_p_by_q = np.mean(log_p)
    expectation_log_q_by_q = np.mean(log_q)
    elbo = expectation_log_p_by_q - expectation_log_q_by_q
    
    
    return elbo
