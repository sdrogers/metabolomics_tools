from collections import Counter
from collections import defaultdict
from random import shuffle
import sys
import time
import math

import numpy as np
import scipy.sparse as sp


class DpMixtureGibbs:
    
    def __init__(self, data, hyperpars):
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
        
        self.Z = None
        self.ZZ_all = sp.lil_matrix((self.N, self.N),dtype=np.float)
        self.cluster_rt_mean = None
        self.cluster_rt_prec = None
        self.matching_results = []
        self.samples_obtained = 0

                
    def run(self):
        
        # initialise all rows under one cluster
        K = 1
        cluster_counts = np.array([float(self.N)])
        cluster_sums = np.array([self.rts.sum()])
        current_ks = np.zeros(self.N, dtype=np.int)

        # start sampling
        self.samples_obtained = 0
        for s in range(self.nsamps):
            
            start_time = time.time()
            
            # loop through the objects in random order
            random_order = range(self.N)
            shuffle(random_order)
            for n in random_order:

                current_data = self.rts[n]
                k = current_ks[n] # the current cluster of this item
                
                # remove from model, detecting empty table if necessary
                cluster_counts[k] = cluster_counts[k] - 1
                cluster_sums[k] = cluster_sums[k] - current_data
                
                # if empty table, delete this cluster
                if cluster_counts[k] == 0:
                    K = K - 1
                    cluster_counts = np.delete(cluster_counts, k) # delete k-th entry
                    cluster_sums = np.delete(cluster_sums, k) # delete k-th entry
                    current_ks = self.reindex(k, current_ks) # remember to reindex all the clusters
                    
                # compute prior probability for K existing table and new table
                prior = np.array(cluster_counts)
                prior = np.append(prior, self.alpha)
                prior = prior / prior.sum()
                
                # for current k
                param_beta = self.rt_prior_prec + (self.rt_prec*cluster_counts)
                temp = (self.rt_prior_prec*self.mu_zero) + (self.rt_prec*cluster_sums)
                param_alpha = (1/param_beta)*temp
                
                # for new k
                param_beta = np.append(param_beta, self.rt_prior_prec)
                param_alpha = np.append(param_alpha, self.mu_zero)
                
                # pick new k
                prec = 1/((1/param_beta)+(1/self.rt_prec))
                log_likelihood = -0.5*np.log(2*np.pi)
                log_likelihood = log_likelihood + 0.5*np.log(prec)
                log_likelihood = log_likelihood - 0.5*np.multiply(prec, np.square(current_data-param_alpha))
                
                # Only bins that have to same top_ids can be together
                this_bin = self.bins[n]
                for kk in range(K):
                    # find which items in cluster kk
                    pos = np.flatnonzero(current_ks==kk)
                    pos = pos.tolist()
                    tops = [self.bins[binpos].top_id for binpos in pos]
                    valid_cluster = True
                    for tt in tops:
                        if tt != this_bin.top_id:
                            valid_cluster = False
                            break
                    if not valid_cluster:
                        log_likelihood[kk] = float('-inf')
                
                # sample from posterior
                post = log_likelihood + np.log(prior)
                post = np.exp(post - post.max())
                post = post / post.sum()
                random_number = np.random.rand()
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
                    cluster_sums = np.append(cluster_sums, current_data)
                else:
                    # put into existing cluster
                    cluster_counts[new_k] = cluster_counts[new_k] + 1
                    cluster_sums[new_k] = cluster_sums[new_k] + current_data

                # assign object to the cluster new_k, regardless whether it's current or new
                current_ks[n] = new_k 

                assert len(cluster_counts) == K, "len(cluster_counts)=%d != K=%d)" % (len(cluster_counts), K)
                assert len(cluster_sums) == K, "len(cluster_sums)=%d != K=%d)" % (len(cluster_sums), K)                    
                assert current_ks[n] < K, "current_ks[%d] = %d >= %d" % (n, current_ks[n])
        
            # end objects loop
            
            time_taken = time.time() - start_time
            if s >= self.burn_in:
            
                print('\tSAMPLE\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))
                self.Z = self.get_Z(self.N, K, current_ks)
                self.ZZ_all = self.ZZ_all + self.get_ZZ(self.Z)
                self.samples_obtained += 1
            
                # construct the actual alignment here
                for k in range(K):
                    pos = np.flatnonzero(current_ks==k)
                    members = [self.bins[a] for a in pos.tolist()]
                    results = []
                    if len(members) == 1:
                        # just singleton things
                        features = members[0].features
                        for f in features:
                            tup = (f, )
                            results.append(tup)                        
                    else:
                        # need to match across the same bins
                        processed = set()
                        for bb1 in members:
                            features1 = bb1.features
                            for f1 in features1:
                                if f1 in processed:
                                    continue
                                # find features in other bins that are the closest in mass to f1
                                temp = []
                                temp.append(f1)
                                processed.add(f1)
                                for bb2 in members:
                                    if bb1.origin == bb2.origin:
                                        continue
                                    else:
                                        features2 = bb2.features
                                        closest = None
                                        min_diff = float('inf')
                                        for f2 in features2:
                                            if f2 in processed:
                                                continue
                                            diff = abs(f1.mass - f2.mass)
                                            if diff < min_diff:
                                                min_diff = diff
                                                closest = f2
                                        if closest is not None:
                                            temp.append(closest)
                                            processed.add(closest)
                                tup = tuple(temp)
                                results.append(tup)    
                    self.matching_results.extend(results)                     
                            
            else:
                print('\tBURN-IN\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))                
            sys.stdout.flush()
                        
        # end sample loop
        
        self.ZZ_all = self.ZZ_all / self.samples_obtained
        print "DONE!"
        
    def reindex(self, deleted_k, current_ks):
        pos = np.where(current_ks > deleted_k)
        current_ks[pos] = current_ks[pos] - 1
        return current_ks
    
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
        
class DpMixtureVariational:
    
    def __init__(self, data, hyperpars):
        ''' 
        Clusters bins by DP mixture model using variational inference
        '''
        print 'DpMixtureVariational initialised'
        
    def run(self):
        print "Hello"