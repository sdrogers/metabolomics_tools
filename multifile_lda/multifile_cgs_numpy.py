import sys

from scipy.special import gammaln
from multifile_utils import estimate_alpha_from_counts

import numpy as np
import pylab as plt

def sample_numpy(random_state, n_burn, n_samples, n_thin, 
            F, Ds, N, K, document_indices, 
            alphas, beta, Z,
            cdk, cd, ckn, ck):    

    N_beta = np.sum(beta)
    for samp in range(n_samples):
    
        s = samp+1        
        if s >= n_burn:
            print("Sample " + str(s) + " ")
        else:
            print("Burn-in " + str(s) + " ")

        thetas = []
        posterior_alphas = []            
        for f in range(F):
            
            print " - file " + str(f) + " ",
            file_D = Ds[f]
            file_cdk = cdk[f]
            file_ckn = ckn[f]
            file_ck = ck[f]
            file_alpha = alphas[f]
            
            for d in range(file_D):
    
                if d%10==0:                        
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                word_locs = document_indices[(f, d)]
                for pos, n in word_locs:
                    
                    # remove word from model
                    # print (f, d, pos)
                    k = Z[(f, d, pos)]
                    
                    file_cdk[d, k] -= 1
                    file_ckn[k, n] -= 1
                    file_ck[k] -= 1
                    
                    all_files_ck = np.zeros_like(file_ck)
                    for this_f in range(F):
                        all_files_ck += ck[this_f]
    
                    log_prior = np.log(file_cdk[d, :] + file_alpha)    
                    log_likelihood = np.log(file_ckn[:, n] + beta[n]) - np.log(all_files_ck + N_beta)

                    # sample new k from the posterior distribution log_post
                    log_post = log_likelihood + log_prior
                    post = np.exp(log_post - log_post.max())
                    post = post / post.sum()
                    
                    # k = random_state.multinomial(1, post).argmax()
                    cumsum = np.empty(K, dtype=np.float64)
                    random_number = random_state.rand()                                
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
             
                    # reassign word back into model
                    file_cdk[d, k] += 1
                    file_ckn[k, n] += 1
                    file_ck[k] += 1
                    Z[(f, d, pos)] = k
            print
            
            # update theta for this file
            theta = file_cdk + file_alpha 
            theta /= np.sum(theta, axis=1)[:, np.newaxis]
            thetas.append(theta)
            
            # update alpha for this file
            alpha_new = estimate_alpha_from_counts(file_D, K, file_alpha, file_cdk)
            alphas[f] = alpha_new
            posterior_alphas.append(alpha_new)

        # update phi for all files
        all_files_ckn = None
        for f in range(F):
            if all_files_ckn is None:
                all_files_ckn = ckn[f]
            else:
                all_files_ckn += ckn[f]
        phi = all_files_ckn + beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
            
#         if s > n_burn:
#         
#             thin += 1
#             if thin%n_thin==0:    
#                 
#                 ll = 0
#                 ll += p_w_z(F, N, K, beta, N_beta, ckn, ck)
#                 for ll_f in range(F):
#                     f_D = Ds[ll_f]
#                     f_alpha = alphas[ll_f]
#                     f_cdk = cdk[ll_f]
#                     f_cd = cd[ll_f]
#                     f_K_alpha = np.sum(f_alpha)
#                     ll += p_z(f_D, K, f_alpha, f_K_alpha, f_cdk, f_cd)                  
#                 all_lls.append(ll)      
#                 print(" - Log likelihood = %.3f " % ll)                                          

    return phi, thetas, posterior_alphas

def p_w_z(F, N, K, beta, N_beta, ckn, ck):

    all_files_ckn = None
    all_files_ck = None
    for f in range(F):
        if all_files_ckn is None:
            all_files_ckn = ckn[f]
            all_files_ck = ck[f]
        else:
            all_files_ckn += ckn[f]
            all_files_ck += ck[f]
    
    val = K * ( gammaln(N_beta) - np.sum(gammaln(beta)) )
    for k in range(K):
        for n in range(N):
            val += gammaln(all_files_ckn[k, n]+beta[n])
        val -= gammaln(all_files_ck[k] + N_beta)      

    return val

def p_z(D, K, alpha, K_alpha, cdk, cd):
    val = D * ( gammaln(K_alpha) - np.sum(gammaln(alpha)) )
    for d in range(D):
        for k in range(K):
            val += gammaln(cdk[d, k]+alpha[k])
        val -= gammaln(cd[d] + K_alpha)                
    return val