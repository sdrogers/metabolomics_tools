import sys
from random import shuffle
import time

from collections import Counter
import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState
from discretisation import utils as DiscretisationUtils

try:
    from dp_rt_sample_index_numba import get_new_k
    print "Numba get_new_k used"
except Exception:
    from dp_rt_sample_index_numpy import get_new_k    
    print "Numpy get_new_k used"
# from dp_rt_sample_index_numpy import get_new_k    

class DpMixtureGibbs:
    
    def __init__(self, data, hyperpars, seed=-1):
        
        ''' 
        Clusters bins by DP mixture model using Gibbs sampling
        '''
        print 'DpMixtureGibbs initialised'
        self.rts = np.array(data[0])
        self.bins = data[1]
        self.N = len(self.rts)
        self.mu_zero = np.mean(self.rts)
        self.rt_prec = float(hyperpars.rt_prec)
        self.rt_prior_prec = float(hyperpars.rt_prior_prec)
        self.alpha = float(hyperpars.alpha)
        self.nsamps = 200
        self.burn_in = 100
        if seed > 0:
            self.random_state = RandomState(seed)
        else:
            self.random_state = RandomState()
        
        self.Z = None
        self.ZZ_all = sp.lil_matrix((self.N, self.N),dtype=np.float)
        self.cluster_rt_mean = None
        self.cluster_rt_prec = None
        self.matching_results = []
        self.samples = []
        self.verbose = False
                
    def run(self):
        
        # initialise all rows under one cluster
        K = 1
        cluster_counts = np.array([float(self.N)])
        cluster_sums = np.array([self.rts.sum()])
        current_ks = np.zeros(self.N, dtype=np.int)
        cluster_member_topids = []
        cluster_member_origins = []

        # everything assigned to the first cluster
        topids = []
        origins = []
        for bb in self.bins:
            topids.append(bb.top_id)
            origins.append(bb.origin)
        cluster_member_topids.append(topids[0])
        cluster_member_origins.append(Counter())
        cluster_member_origins[0].update(origins)

        # start sampling
        for s in range(self.nsamps):
            
            start_time = time.time()
            
            # loop through the objects in random order
            random_order = range(self.N)
            self.random_state.shuffle(random_order)
            for n in random_order:

                this_bin = self.bins[n]
                current_data = self.rts[n]
                k = current_ks[n] # the current cluster of this item

                start = time.time()
        
                # remove from model, detecting empty table if necessary
                cluster_counts[k] = cluster_counts[k] - 1
                cluster_sums[k] = cluster_sums[k] - current_data
                cluster_member_origins[k].subtract([this_bin.origin])

                end = time.time()
                if self.verbose:
                    DiscretisationUtils.timer("Remove from model", start, end)    
                
                # if empty table, delete this cluster
                if cluster_counts[k] == 0:
                    start = time.time()
                    K = K - 1
                    # remove k-th entry
                    cluster_counts = np.delete(cluster_counts, k)
                    cluster_sums = np.delete(cluster_sums, k)
                    del cluster_member_topids[k]
                    del cluster_member_origins[k]
                    current_ks = self.reindex(k, current_ks) # remember to reindex all the clusters
                    end = time.time()
                    if self.verbose:
                        DiscretisationUtils.timer("Clear empty table", start, end)    
                    
                # check the valid cluster this bin can go into
                start = time.time()
                valid_clusters_check = np.empty(K)
                valid_clusters_check.fill(float('-inf'))
                topids_arr = np.array(cluster_member_topids)
                possible = np.where(topids_arr == this_bin.top_id)[0]
                for k_idx in possible:
                    # can only go into a cluster where there is no bin with the same origin file
                    existing_origins = cluster_member_origins[k_idx]
                    if existing_origins[this_bin.origin] == 0:
                        valid_clusters_check[k_idx] = 0 # because log(1) = 0

                # get the actual possible mask to use when sampling from the posterior
                possible = np.where(valid_clusters_check == 0)[0]
                        
                end = time.time()
                if self.verbose:                
                    DiscretisationUtils.timer("Check valid cluster", start, end)    

                start = time.time()                    
                # if no possible current cluster, then immediately make a new cluster
                if len(possible) == 0:
                    new_k = K
                else:                
                    # otherwise resample the cluster index
                    random_number = self.random_state.rand()
                    cc = cluster_counts[possible]
                    cs = cluster_sums[possible]
                    # prior for K existing tables + new table
                    prior = np.append(cluster_counts, self.alpha)
                    prior = prior / prior.sum()       
                    log_prior = np.log(prior)
                    last_val = log_prior[-1]
                    log_prior = log_prior[possible]
                    log_prior = np.append(log_prior, last_val)
                    new_k = get_new_k(self.alpha, self.rt_prior_prec, self.rt_prec, self.mu_zero, 
                                      cc, cs, log_prior, current_data, possible, random_number, K)
                end = time.time()
                if self.verbose:                
                    DiscretisationUtils.timer("Get new k", start, end)    

                start = time.time()                                                            
                # (new_k+1) because indexing starts from 0 here
                if (new_k+1) > K:
                    # make new cluster and add to it
                    K = K + 1
                    cluster_counts = np.append(cluster_counts, 1)
                    cluster_sums = np.append(cluster_sums, current_data)
                    cluster_member_topids.append(this_bin.top_id)
                    c = Counter([this_bin.origin])
                    cluster_member_origins.append(c)
                else:
                    # put into existing cluster
                    cluster_counts[new_k] = cluster_counts[new_k] + 1
                    cluster_sums[new_k] = cluster_sums[new_k] + current_data
                    assert cluster_member_topids[new_k] == this_bin.top_id
                    cluster_member_origins[new_k].update([this_bin.origin])
                end = time.time()
                if self.verbose:
                    DiscretisationUtils.timer("Assign to cluster", start, end)    

                # assign object to the cluster new_k, regardless whether it's current or new
                current_ks[n] = new_k 

                assert len(cluster_counts) == K, "len(cluster_counts)=%d != K=%d)" % (len(cluster_counts), K)
                assert len(cluster_sums) == K, "len(cluster_sums)=%d != K=%d)" % (len(cluster_sums), K)                    
                assert len(cluster_member_topids) == K, "len(cluster_member_topids)=%d != K=%d)" % (len(cluster_member_topids), K)                    
                assert len(cluster_member_origins) == K, "len(cluster_member_origins)=%d != K=%d)" % (len(cluster_member_origins), K)
                assert current_ks[n] < K, "current_ks[%d]=%d but K=%d" % (n, current_ks[n], K)
        
            # end objects loop
            
            time_taken = time.time() - start_time
            if s >= self.burn_in:
                start = time.time()                                                                                        
                print('\tSAMPLE\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))
                self.store_sample(K, current_ks)
            
                # construct the actual alignment here
                for k in range(K):
                    pos = np.flatnonzero(current_ks==k)
                    members = [self.bins[a] for a in pos.tolist()]
                    memberstup = tuple(members)
                    print "sample=" + str(s) + " k=" + str(k) + " memberstup=" + str(memberstup)
                    self.matching_results.append(memberstup)
                end = time.time()
                if self.verbose:                
                    DiscretisationUtils.timer("Store sample", start, end)    
            else:
                print('\tBURN-IN\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))                
            sys.stdout.flush()
                        
        # end sample loop
        
        self.samples_obtained = len(self.samples)
        counter = 0
        for samp in self.samples:
            print('\tProcessing sample %d' % (counter))
            K = samp[0]
            current_ks = samp[1]
            self.Z = self.get_Z(self.N, K, current_ks)
            self.ZZ_all = self.ZZ_all + self.get_ZZ(self.Z)
            counter += 1
        self.ZZ_all = self.ZZ_all / self.samples_obtained
        print "DONE!"
                
    def reindex(self, deleted_k, current_ks):
        pos = np.where(current_ks > deleted_k)
        current_ks[pos] = current_ks[pos] - 1
        return current_ks
    
    def store_sample(self, K, current_ks):
        sample = (K, current_ks.copy())
        self.samples.append(sample)
    
    def get_Z(self, N, K, current_ks):
        Z = sp.lil_matrix((N, K))
        for n in range(len(current_ks)):
            k = current_ks[n]
            Z[n, k] = 1
        return Z
    
    def get_ZZ(self, Z):
        return Z.tocsr() * Z.tocsr().transpose()
    
    def __repr__(self):
        return "Gibbs sampling for DP mixture model\n" + self.hyperpars.__repr__() + \
        "\nn_samples = " + str(self.n_samples)