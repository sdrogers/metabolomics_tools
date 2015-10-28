import numpy as np

def get_new_k(alpha, rt_prior_prec, rt_prec, mu_zero, 
                      cluster_counts, cluster_sums, log_prior, current_data,
                      possible, random_number, K):
            
    # for current k
    param_beta = rt_prior_prec + (rt_prec*cluster_counts)
    temp = (rt_prior_prec*mu_zero) + (rt_prec*cluster_sums)
    param_alpha = (1/param_beta)*temp
    
    # for new k
    param_beta = np.append(param_beta, rt_prior_prec)
    param_alpha = np.append(param_alpha, mu_zero)
    
    # pick new k
    prec = 1/((1/param_beta)+(1/rt_prec))
    log_likelihood = -0.5*np.log(2*np.pi)
    log_likelihood = log_likelihood + 0.5*np.log(prec)
    log_likelihood = log_likelihood - 0.5*np.multiply(prec, np.square(current_data-param_alpha))                
    
    # sample from posterior
    post = log_likelihood + log_prior
    post = np.exp(post - post.max())
    post = post / post.sum()
    cumsum = np.cumsum(post)
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