from numpy.random import RandomState
import operator
import sys
import time

from scipy.special import gammaln

from models import HyperPars
import numpy as np
import scipy.sparse as sp


class DpMixtureGibbs:
    
    def __init__(self, data, hyperpars, seed=-1, verbose=False):
        ''' 
        Clusters bins by DP mixture model using Gibbs sampling
        '''
        self.verbose = verbose
        if self.verbose:
            print 'DpMixtureGibbs initialised'
            sys.stdout.flush()
            
        self.use_rt_likelihood = hyperpars.second_stage_clustering_use_rt_likelihood
        self.use_mass_likelihood = hyperpars.second_stage_clustering_use_mass_likelihood
        self.use_adduct_likelihood = hyperpars.second_stage_clustering_use_adduct_likelihood

        # prepare arrays for concrete bins, posterior rts 
        # and word counts (from 1st stage clustering)
        self.masses = np.array(data[0])
        self.rts = np.array(data[1])
        self.word_counts_list = [np.array(x) for x in data[2]]
        
        self.W = len(self.word_counts_list[0])
        self.origins = data[3]
        self.N = len(self.rts)
        assert self.N == len(self.word_counts_list)
        assert self.N == len(self.origins)

        delta = hyperpars.across_file_rt_tol
        var = (delta/3.0)**2 # assume 1 delta is 3 standard deviations
        self.rt_prec = 1.0/var

        log_one_ppm = np.log(1000001) - np.log(1000000)
        log_delta = log_one_ppm * hyperpars.across_file_mass_tol
        delta = np.exp(log_delta)
        var = (delta/3.0)**2 # assume 1 delta is 3 standard deviations
        self.mass_prec = 1.0/var        

        self.prior_rt_mean = np.mean(self.rts)
        self.prior_rt_prec = 5E-6

        self.prior_mass_mean = np.mean(self.masses)
        self.prior_mass_prec = 5E-6
        
        self.alpha = float(hyperpars.dp_alpha)
        self.beta = float(hyperpars.beta)
        
        self.nsamps = hyperpars.rt_clustering_nsamps
        self.burn_in = hyperpars.rt_clustering_burnin
        self.seed = int(seed)
        if self.seed > 0:
            self.random_state = RandomState(self.seed)
        else:
            self.random_state = RandomState()        
        
        # self.Z = None
        # self.ZZ_all = sp.lil_matrix((self.N, self.N),dtype=np.float)
        self.cluster_rt_mean = None
        self.cluster_rt_prec = None
        self.matching_results = []
        self.samples_obtained = 0

                
    def run(self):

        if self.verbose:
            print "Sampling begins"
            sys.stdout.flush()
        
        # initialise all rows under one cluster
        K = 1
        cluster_counts = np.array([float(self.N)])
        cluster_mass_sums = np.array([self.masses.sum()])
        cluster_rt_sums = np.array([self.rts.sum()])
        all_word_counts = np.zeros(self.W)
        for wc in self.word_counts_list:
            all_word_counts += wc
        cluster_word_sums = [all_word_counts]
        current_ks = np.zeros(self.N, dtype=np.int)
        cluster_member_origins = [list(self.origins)]

        # start sampling
        self.samples_obtained = 0
        for s in range(self.nsamps):
            
            start_time = time.time()
            
            if self.N > 1: # if only 1 item, then nothing to sample
                
                # loop through the objects in random order
                random_order = range(self.N)
                self.random_state.shuffle(random_order)
                processed = 0
                for n in random_order:
                    
                    if self.verbose:
                        processed += 1
                        print "Processing %s remaining %d/%d" % ((s, n), processed, self.N) 
                        sys.stdout.flush()

                    current_mass_data = self.masses[n]                                    
                    current_rt_data = self.rts[n]
                    current_word_counts = self.word_counts_list[n]
                    current_origin = self.origins[n]
                    k = current_ks[n] # the current cluster of this item
                    
                    # remove from model, detecting empty table if necessary
                    cluster_counts[k] = cluster_counts[k] - 1
                    cluster_mass_sums[k] = cluster_mass_sums[k] - current_mass_data
                    cluster_rt_sums[k] = cluster_rt_sums[k] - current_rt_data
                    cluster_word_sums[k] = cluster_word_sums[k] - current_word_counts
                    cluster_member_origins[k].remove(current_origin)
                    
                    # if empty table, delete this cluster
                    if cluster_counts[k] == 0:
                        K = K - 1
                        cluster_counts = np.delete(cluster_counts, k) # delete k-th entry
                        cluster_mass_sums = np.delete(cluster_mass_sums, k) # delete k-th entry
                        cluster_rt_sums = np.delete(cluster_rt_sums, k) # delete k-th entry
                        del cluster_member_origins[k] # delete k-th entry
                        del cluster_word_sums[k]
                        current_ks = self._reindex(k, current_ks) # remember to reindex all the clusters
                        
                    # compute prior probability for K existing table and new table
                    prior = np.array(cluster_counts)
                    prior = np.append(prior, self.alpha)
                    prior = prior / prior.sum()

                    log_likelihood = np.zeros_like(prior)

                    ## mass likelihood
                    if self.use_mass_likelihood:
                    
                        # for current k
                        param_beta = self.prior_mass_prec + (self.mass_prec*cluster_counts)
                        temp = (self.prior_mass_prec*self.prior_mass_mean) + (self.mass_prec*cluster_mass_sums)
                        param_alpha = (1/param_beta)*temp
                        
                        # for new k
                        param_beta = np.append(param_beta, self.prior_mass_prec)
                        param_alpha = np.append(param_alpha, self.prior_mass_mean)
                        
                        # pick new k
                        prec = 1/((1/param_beta)+(1/self.mass_prec))
                        log_likelihood_mass = -0.5*np.log(2*np.pi)
                        log_likelihood_mass = log_likelihood_mass + 0.5*np.log(prec)
                        log_likelihood_mass = log_likelihood_mass - 0.5*np.multiply(prec, np.square(current_mass_data-param_alpha))

                        log_likelihood += log_likelihood_mass
                        self.like_mass = log_likelihood_mass
                    
                    ## RT likelihood
                    if self.use_rt_likelihood:
                    
                        # for current k
                        param_beta = self.prior_rt_prec + (self.rt_prec*cluster_counts)
                        temp = (self.prior_rt_prec*self.prior_rt_mean) + (self.rt_prec*cluster_rt_sums)
                        param_alpha = (1/param_beta)*temp
                        
                        # for new k
                        param_beta = np.append(param_beta, self.prior_rt_prec)
                        param_alpha = np.append(param_alpha, self.prior_rt_mean)
                        
                        # pick new k
                        prec = 1/((1/param_beta)+(1/self.rt_prec))
                        log_likelihood_rt = -0.5*np.log(2*np.pi)
                        log_likelihood_rt = log_likelihood_rt + 0.5*np.log(prec)
                        log_likelihood_rt = log_likelihood_rt - 0.5*np.multiply(prec, np.square(current_rt_data-param_alpha))

                        log_likelihood += log_likelihood_rt
                        self.like_rt = log_likelihood_rt
    
                    ## adducts likelihood
                    if self.use_adduct_likelihood:
    
                        log_likelihood_wc = np.zeros_like(log_likelihood_rt)
                        for k_idx in range(K): # the finite portion
                            wcb = cluster_word_sums[k_idx] + self.beta
                            log_likelihood_wc[k_idx] = self._C(wcb+current_word_counts) - self._C(wcb)
                        # the infinite bit
                        wcb = np.zeros(self.W) + self.beta
                        log_likelihood_wc[-1] = self._C(wcb+current_word_counts) - self._C(wcb)
    
                        log_likelihood += log_likelihood_wc
                        self.like_wc = log_likelihood_wc
                    
                    ## plus some additional rules to prevent bins from different files to be clustered together                
                    valid_clusters_check = np.zeros(K+1)
                    for k_idx in range(K):
                        # this_bin cannot go into a cluster where the origin file is the same
                        existing_origins = cluster_member_origins[k_idx]
                        if current_origin in existing_origins:
                            valid_clusters_check[k_idx] = float('-inf')
                    log_likelihood = log_likelihood + valid_clusters_check                
                    
                    # sample from posterior
                    post = log_likelihood + np.log(prior)
                    post = np.exp(post - post.max())
                    post = post / post.sum()
                    random_number = self.random_state.rand()
                    cumsum = np.cumsum(post)
                    new_k = 0
                    for new_k in range(len(cumsum)):
                        c = cumsum[new_k]
                        if random_number <= c:
                            break
                        
                    # (new_k+1) because indexing starts from 0 here
                    if (new_k+1) > K:
                        # make new cluster and add to it
                        K = K + 1
                        cluster_counts = np.append(cluster_counts, 1)
                        cluster_mass_sums = np.append(cluster_mass_sums, current_mass_data)
                        cluster_rt_sums = np.append(cluster_rt_sums, current_rt_data)
                        cluster_member_origins.append([current_origin])
                        cluster_word_sums.append(current_word_counts)
                    else:
                        # put into existing cluster
                        cluster_counts[new_k] = cluster_counts[new_k] + 1
                        cluster_mass_sums[new_k] = cluster_mass_sums[new_k] + current_mass_data
                        cluster_rt_sums[new_k] = cluster_rt_sums[new_k] + current_rt_data
                        cluster_member_origins[new_k].append(current_origin)
                        cluster_word_sums[new_k] = cluster_word_sums[new_k] + current_word_counts
    
                    # assign object to the cluster new_k, regardless whether it's current or new
                    current_ks[n] = new_k 
    
                    assert len(cluster_counts) == K, "len(cluster_counts)=%d != K=%d)" % (len(cluster_counts), K)
                    assert len(cluster_mass_sums) == K, "len(cluster_mass_sums)=%d != K=%d)" % (len(cluster_mass_sums), K)                    
                    assert len(cluster_rt_sums) == K, "len(cluster_rt_sums)=%d != K=%d)" % (len(cluster_rt_sums), K)                    
                    assert len(cluster_member_origins) == K, "len(cluster_member_origins)=%d != K=%d)" % (len(cluster_member_origins), K)
                    assert len(cluster_word_sums) == K, "len(cluster_word_sums)=%d != K=%d)" % (len(cluster_word_sums), K)
                    assert current_ks[n] < K, "current_ks[%d] = %d >= %d" % (n, current_ks[n])
            
                # end objects loop
            
            time_taken = time.time() - start_time
            if s >= self.burn_in:
            
                if self.verbose:
                    print('\tSAMPLE\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))
                # self.Z = self._get_Z(self.N, K, current_ks)
                self.samples_obtained += 1
            
                # construct the actual alignment here
                for k in range(K):
                    pos = np.flatnonzero(current_ks==k)
                    memberstup = tuple(pos.tolist())
                    # if self.verbose:
                    #    print "\t\tsample=" + str(s) + " k=" + str(k) + " memberstup=" + str(memberstup)                    
                    self.matching_results.append(memberstup)
            else:
                if self.verbose:
                    print('\tBURN-IN\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))                
            sys.stdout.flush()
                        
        # end sample loop
        self.last_K = K        
        self.last_assignment = current_ks
        if self.verbose:
            print "DONE!"
            
    def _C(self, arr):
        sum_arr = np.sum(arr)
        sum_log_gamma = np.sum(gammaln(arr))
        res = sum_log_gamma - gammaln(sum_arr)
        return res
        
    def _reindex(self, deleted_k, current_ks):
        pos = np.where(current_ks > deleted_k)
        current_ks[pos] = current_ks[pos] - 1
        return current_ks
    
#     def _get_Z(self, N, K, current_ks):
#         Z = sp.lil_matrix((N, K))
#         for n in range(len(current_ks)):
#             k = current_ks[n]
#             Z[n, k] = 1
#         return Z
#     
#     def _get_ZZ(self, Z):
#         return Z.tocsr() * Z.tocsr().transpose()
    
    def __repr__(self):
        return "Gibbs sampling for DP mixture model\n" + self.hyperpars.__repr__() + \
        "\nn_samples = " + str(self.n_samples)
        
def main(argv):    

    # just a simple test case
    masses = [100, 100, 100, 100, 100, 100.1]
    rts = [10, 11, 12, 100, 101, 98]
    word_counts = [
                   [0, 0, 0, 10, 0, 90],        # item 0
                   [0, 0, 0, 10, 0, 90],        # item 1
                   [0, 0, 0, 90, 0, 10],        # item 2
                   [0, 0, 0, 25, 0, 75],        # item 3
                   [0, 0, 0, 3, 0, 97],         # item 4
                   [0, 0, 0, 4, 0, 96],         # item 5
                   ]
    origins = [1, 2, 3, 4, 5, 6]
    
    data = (masses, rts, word_counts, origins)

    hp = HyperPars()
    hp.rt_clustering_nsamps = 200
    hp.rt_clustering_burnin = 100
    hp.dp_alpha = 100
    clusterer = DpMixtureGibbs(data, hp, seed=1234567890, verbose=True) 
    clusterer.run()   
    
    results = {}
    for item in clusterer.matching_results:
        if item in results:
            results[item] += 1
        else:
            results[item] = 1

    sorted_results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)    
    for key, value in sorted_results:
        print "%s = %d" % (key, value)
        
if __name__ == "__main__":
   main(sys.argv[1:])
