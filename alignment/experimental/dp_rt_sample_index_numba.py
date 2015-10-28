import numpy as np
import math
from numba import jit

@jit(nopython=True)
def append_last(array, scalar):
    arr_len = len(array)
    new_arr = np.zeros(arr_len+1)
    for i in range(len(array)):
        new_arr[i] = array[i]
    new_arr[i+1] = scalar
    return new_arr

@jit(nopython=True)
def get_new_k(alpha, rt_prior_prec, rt_prec, mu_zero, 
                      cluster_counts, cluster_sums, log_prior, current_data,
                      possible, random_number, K):
        
    # for current k
    param_beta = rt_prior_prec + (rt_prec*cluster_counts)
    # temp = (rt_prior_prec*mu_zero) + (rt_prec*cluster_sums)
    temp = np.zeros(len(cluster_sums))
    for k in range(len(cluster_sums)):
        temp[k] = (rt_prior_prec*mu_zero) + (rt_prec*cluster_sums[k])
    param_alpha = (1/param_beta)*temp
    
    # for new k
    param_beta = append_last(param_beta, rt_prior_prec)
    param_alpha = append_last(param_alpha, mu_zero)
    
    # pick new k
    # prec = 1/((1/param_beta)+(1/rt_prec))
    prec = np.zeros(len(param_beta))
    for k in range(len(param_beta)):
        prec[k] = 1/((1/param_beta[k])+(1/rt_prec))
    temp1 = -0.5*np.log(2*np.pi)
    temp2 =  0.5*np.log(prec)
    temp3 = np.zeros(len(prec))
    for k in range(len(prec)):        
        subs = current_data-param_alpha[k]
        temp3[k] = -(0.5*prec[k] * (subs*subs))
    log_likelihood = temp1 + temp2 + temp3
    
    # sample from posterior
    post = log_likelihood + log_prior
    cumsum = np.zeros(len(post))

    # post = np.exp(post - post.max())
    max_log_post = post[0]
    for i in range(len(post)):
        val = post[i]
        if val > max_log_post:
            max_log_post = val

    # post = post / post.sum()
    sum_post = 0
    for i in range(len(post)):
        post[i] = math.exp(post[i] - max_log_post)
        sum_post += post[i]
    for i in range(len(post)):
        post[i] = post[i] / sum_post

    # k = np.random.multinomial(1, post).argmax()
    total = 0
    for i in range(len(post)):
        val = post[i]
        total += val
        cumsum[i] = total
    k = 0
    for k in range(len(cumsum)):
        c = cumsum[k]
        if random_number <= c:
            break 

    if k+1 > len(possible):
        # need to create a new cluster
        new_k = K
    else:       
        # reuse existing cluster 
        new_k = possible[k]
    return new_k