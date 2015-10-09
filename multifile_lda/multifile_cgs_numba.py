import sys

from scipy.special import gammaln

import numpy as np

def sample_numba(random_state, n_burn, n_samples, n_thin, 
            F, Ds, N, K, document_indices, 
            alphas, beta, Z,
            cdk, cd, ckn, ck):    

    print "Hello world"
