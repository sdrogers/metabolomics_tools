import os
import os.path
import sys
from random import shuffle
import time

from scipy.sparse import *
import scipy.io
import numpy as np

from models import *

class GreedyClustering:

    def __init__(self, alignment_file, options):
        
        self.alignment_file = alignment_file
        self.options = options
        self.grt = float(self.options.grt)  
        self.min_corr_signal = float(self.options.mcs)

    def do_clustering(self):
        
        print " - Grouping " + self.alignment_file.filename
        sys.stdout.flush()
        
        if self.options.use_peakshape:
            tol = '%.1f' % self.grt
            mcs = '%.2f' % self.min_corr_signal
            front_part, extension = os.path.splitext(self.alignment_file.filename)
            mat_path = self.alignment_file.parent_dir + '/../mat'
            filename = front_part + '.greedy_peakshape.' + tol + '_' + mcs + '.mat'
            target = os.path.join(mat_path, filename)
            print target
            if os.path.isfile(target):
                print "\tReading peak shape clustering from " + target
                mdict = scipy.io.loadmat(target)
                ZZ_all = mdict['ZZ_all']
                return ZZ_all
        
        rows = sorted(self.alignment_file.rows, key=lambda x: x.get_average_intensity(), reverse=True)
        N = len(rows)

        groups = []
        for reference_row in rows:
            if reference_row.grouped == True:
                continue
            candidates = self.alignment_file.get_ungrouped_rows(reference_row, self.grt)
            if len(candidates) > 0:
                for candidate_row in candidates:
                    candidate_row.grouped = True
                group = candidates
                groups.append(group)

        K = len(groups)
        Z = lil_matrix((N, K))        

        k = 0
        for group in groups:
            for row in group:
                n = row.row_id
                Z[n, k] = 1
            k = k + 1

        print " - " + str(k) + " groups obtained"
                
        ZZ = Z.tocsr() * Z.tocsr().transpose()
        return ZZ.tolil()

        return ZZ
                
class MixtureModelClustering:
        
    def __init__(self, alignment_file, options):
        
        self.alignment_file = alignment_file
        self.options = options
        self.grt = float(self.options.grt)  
        self.nsamps = int(self.options.num_samples)
        self.burn_in = int(self.options.burn_in)
        self.alpha = float(self.options.dp_alpha)
        self.tau_zero = 5E-3
        self.rt_tol = 1/(self.grt)
        self.data = []
        for row in self.alignment_file.rows:
            self.data.append(row.get_average_rt())
        self.data = np.array(self.data)
        self.N = len(self.data)
        self.mu_zero = np.mean(self.data)

    def do_clustering(self):
        
        print " - Grouping " + self.alignment_file.filename
        sys.stdout.flush()
        
        mat_path = self.alignment_file.parent_dir + '/../mat'
        target = os.path.join(mat_path, self.alignment_file.filename + '.mixture_model_rt.mat')
        if os.path.isfile(target) and not self.options.always_recluster:
            print "\tReading clustering from " + target
            mdict = scipy.io.loadmat(target)
            ZZ_all = mdict['ZZ_all']
            return ZZ_all

        # matfile = os.path.join(self.matlab_dir, self.alignment_file.filename + '.ZZprob')
        # print matfile
        # x = scipy.io.loadmat(matfile)
        # loaded = x['ZZprob']
        # grouping_arr = lil_matrix(loaded)                

        N = self.N
        data = self.data
        ZZ_all = csr_matrix((N, N))

        # initialise all rows under one cluster
        K = 1 # total number of clusters        
        # count of members in each clusters 
        cluster_counts = np.array([float(N)]) # must store float for correct division in prior calculation later
        cluster_sums = np.array([data.sum()])
        current_ks = np.zeros(N, dtype=np.int)

        # start sampling
        total_nsamps = self.nsamps + self.burn_in                
        for s in range(total_nsamps):
            
            start_time = time.time()
            
            # loop through the objects in random order
            random_order = range(N)
            # shuffle(random_order)
            for n in random_order:

                # this is the currently selected object from the random permutation
                current_data = data[n]
                k = current_ks[n] # the current cluster of this row
                current_rt = data[n]
                
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
                param_beta = self.tau_zero + (self.rt_tol*cluster_counts)
                temp = (self.tau_zero*self.mu_zero) + (self.rt_tol*cluster_sums)
                param_alpha = (1/param_beta)*temp
                
                # for new k
                param_beta = np.append(param_beta, self.tau_zero)
                param_alpha = np.append(param_alpha, self.mu_zero)
                
                # pick new k
                prec = 1/((1/param_beta)+(1/self.rt_tol))
                log_likelihood = -0.5*np.log(2*np.pi)
                log_likelihood = log_likelihood + 0.5*np.log(prec)
                log_likelihood = log_likelihood - 0.5*np.multiply(prec, np.square(current_data-param_alpha))
                
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
                Z = self.get_Z(N, K, current_ks)
                ZZ_all = ZZ_all + self.get_ZZ(Z)
            else:
                print('\tBURN-IN\tIteration %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))                
            sys.stdout.flush()
            
        # end sample loop
        
        ZZ_all = ZZ_all / self.nsamps
        
        if not self.options.always_recluster:
            print "Saving clustering result to " + target
            mdict = {'ZZ_all' : ZZ_all}
            scipy.io.savemat(target, mdict, do_compression=True)
                
        return ZZ_all
    
    def reindex(self, deleted_k, current_ks):
        pos = np.where(current_ks > deleted_k)
        current_ks[pos] = current_ks[pos] - 1
        return current_ks
    
    def get_Z(self, N, K, current_ks):
        Z = lil_matrix((N, K))
        for n in range(len(current_ks)):
            k = current_ks[n]
            Z[n, k] = 1
        return Z
    
    def get_ZZ(self, Z):
        return Z.tocsr() * Z.tocsr().transpose()
