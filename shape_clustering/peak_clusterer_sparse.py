
import numpy as np
from scipy.special import gammaln
from scipy.stats import beta
from scipy.sparse import coo_matrix,lil_matrix
import time
import itertools
import pylab as plt

class shape_cluster_gibbs_sparse(object):

    def __init__(self,corr_mat,hyper_pars,n_samples = 100,n_burn = 10,infinite = True,track = False,K=30,output = 0,seed=12345):
        self.corr_mat = corr_mat
        self.hyper_pars = hyper_pars

        self.n_samples = n_samples
        self.n_burn = n_burn

        self.infinite = infinite
        self.track = track
        self.K = 30

        self.seed = seed
        np.random.seed(self.seed)

        self.output = output

        if infinite and track:
            print "Warning: track doesn't work in infinite mode, setting it to false"
            track = False


        self.n_peaks = (corr_mat.shape)[0]

        self._create_like_mats()

    def _initialise_clustering(self):
        if self.infinite:
            # Put everything in one clusterer
            self.K = 1
            self.Z = np.zeros(self.n_peaks,dtype=np.int64)
            self.counts = [self.n_peaks]
        else:
            self.Z = np.zeros(self.n_peaks,dtype=np.int64)
            self.counts = np.zeros(self.K).tolist()
            self.counts[0] = self.n_peaks

        self.base_like = self.out_like.sum(axis=0)
        self.base_like += np.log(1-self.hyper_pars.out_prob)*((self.n_peaks-1)-((self.corr_mat>0).sum(axis=0)))
        self.base_like = np.array(self.base_like)[0]
        if self.track:
            self.temp = np.zeros((self.K,self.n_peaks))
            for k in np.arange(self.K-1):
                # self.temp = np.tile(self.out_like.sum(axis=1),(self.K,1))
                self.temp[k+1,:] = self.base_like

            total_in_like = self.in_like.sum(axis=0)
            total_in_like += np.log(1-self.hyper_pars.in_prob)*((self.n_peaks-1)-((self.corr_mat>0).sum(axis=0)))
            self.temp[0,:] = total_in_like


      

        # Creeate output structures
        self.post_sim = lil_matrix((self.n_peaks,self.n_peaks))
        self.all_K = []
        self.all_total_like = []
        self.best_Z = self.Z
        self.best_like = -np.inf

    def _sample(self):
        invec = np.zeros(self.n_peaks)
        outvec = np.zeros(self.n_peaks)
        self.corr_mat = self.corr_mat.tocsr()
        self.in_like = self.in_like.tocsr()
        self.out_like = self.out_like.tocsr()
        inlog = np.log(1-self.hyper_pars.in_prob)
        outlog = np.log(1-self.hyper_pars.out_prob)
        if not self.infinite:
            self.counts = np.log([c + self.hyper_pars.conc_par/self.K for c in self.counts])
        
        for samp in np.arange(self.n_samples):
            if self.output>0 and samp%10==0:
                print "Sample " + str(samp)
                for peak in np.arange(self.n_peaks):
                    this_peak = peak
                    this_cluster = self.Z[this_peak]
                    t0 = time.time()
                    if self.track:
                        # create the in and out vectors
                        ta = time.time()
                        invec = np.tile(np.log(1-self.hyper_pars.in_prob),self.n_peaks)
                        invec[this_peak] = 0
                        inlikerow = self.in_like.getrow(this_peak).tocoo()
                        # outlikerow = self.out_like.getrow(this_peak).tocoo()
                        # outlikerow = self.out_like.getrow(this_peak).nonzero()
                        obs_pos = inlikerow.col
                        invec[obs_pos] = inlikerow.data
                        # invec[obs_pos] = self.in_like[this_peak,:].nonzero()
                        outvec = np.tile(np.log(1-self.hyper_pars.out_prob),self.n_peaks)
                        outvec[this_peak] = 0
                        outvec[obs_pos] = self.out_like.getrow(this_peak).tocoo().data
                        # outvec[obs_pos] = self.out_like[this_peak,:].nonzero()
                        self.temp -= outvec
                        self.temp[this_cluster,:]+=outvec-invec

       
                    
                    self.Z[this_peak] = -1   
                    if self.infinite: 
                        self.counts[this_cluster]-=1  
                    else:
                        self.counts[this_cluster] = np.log(np.exp(self.counts[this_cluster])-1)
                    
                        

                    if self.counts[this_cluster] == 0 and self.infinite:
                        self.K -= 1
                        self.counts.pop(this_cluster)
                        self.Z[np.where(self.Z>this_cluster)] -= 1

         

                    if self.infinite:
                        prior = np.log(np.hstack((self.counts,self.hyper_pars.conc_par)))
                    else:
                        # prior = np.log([c + self.hyper_pars.conc_par/self.K for c in self.counts])
                        prior = self.counts


  
                    if not self.track:
                        like = np.zeros_like(prior)
                        for k in np.arange(self.K):
                            in_pos = np.where(self.Z==k)[0]
                            both = np.intersect1d(in_pos,obs_pos)
                            in_not_obs = np.setdiff1d(in_pos,obs_pos)
                            like[k] = self.base_like[this_peak] - self.out_like[this_peak,both].sum()
                            like[k] -= np.log(1-self.hyper_pars.out_prob)*in_not_obs.size
                            like[k] += self.in_like[this_peak,both].sum()
                            like[k] += np.log(1-self.hyper_pars.in_prob)*in_not_obs.size

                        if self.infinite:
                            like[-1] = self.base_like[this_peak]
                    else:
                        like = self.temp[:,this_peak]


                    post = prior + like
                    post = np.exp(post - np.max(post))
                    post /= post.sum()
                    new_cluster = np.where(np.random.rand()<post.cumsum())[0][0]

                    if new_cluster >= self.K:
                        # This can only happen in an infinite mix
                        self.Z[this_peak] = self.K
                        self.counts.append(1)
                        self.K += 1
                    else:
                        self.Z[this_peak] = new_cluster
                        if self.infinite:
                            self.counts[new_cluster] += 1
                        else:
                            self.counts[new_cluster] = np.log(np.exp(self.counts[new_cluster]) + 1)

                        if self.track:
                            
                            self.temp += outvec
                            self.temp[new_cluster,:] += invec - outvec


            if samp > self.n_burn:
                total_like = self.base_like.sum()
                # self.post_sim += (self.Z[:,np.newaxis] == self.Z)
                for k in np.arange(self.K):
                    pos = np.where(self.Z==k)[0]
                    # self.post_sim[pos[:,np.newaxis],pos] += 1
                    total_like -= self.out_like[pos[:,np.newaxis],pos].sum()
                    total_like += self.in_like[pos[:,np.newaxis],pos].sum()
                    
                self.all_K.append(self.K)
                self.all_total_like.append(total_like)
                if total_like > self.best_like:
                    self.best_like = total_like
                    self.best_Z = self.Z



        self.post_sim /= (self.n_samples - self.n_burn)

    def _summary(self):
        print "K = " + str(self.K)
        plt.figure()
        plt.pcolor(np.array(self.post_sim.todense()))
        plt.figure()
        plt.hist(np.sort(self.counts),np.arange(np.min(self.counts),np.max(self.counts)+1))
        plt.xlabel('Cluster size')
        plt.ylabel('Count')
        if self.infinite:
            plt.figure()
            plt.plot(np.arange(self.n_samples-self.n_burn-1),np.array(self.all_K))
            plt.xlabel('Samples')
            plt.ylabel('K')
        plt.figure()
        plt.plot(np.arange(self.n_samples-self.n_burn-1),np.array(self.all_total_like))
        plt.xlabel('Samples')
        plt.ylabel('Log likelihood')
        print "Best Log Like = " + str(self.best_like) + " (K = " + str(np.max(self.best_Z)+1) + ")"

    def _create_like_mats(self):
        print "Creating likelihood matrices"
        self.in_like = lil_matrix((self.n_peaks,self.n_peaks))
        self.out_like = lil_matrix((self.n_peaks,self.n_peaks))
        cx = self.corr_mat.tocoo()

        for n,m,v in itertools.izip(cx.row,cx.col,cx.data):
            in_val = np.log(self.hyper_pars.in_prob) + log_beta_pdf(v,self.hyper_pars.in_alpha,self.hyper_pars.in_beta)
            out_val = np.log(self.hyper_pars.out_prob) + log_beta_pdf(v,self.hyper_pars.out_alpha,self.hyper_pars.out_beta)
            self.in_like[n,m] = in_val
            self.in_like[m,n] = in_val
            self.out_like[n,m] = out_val
            self.out_like[m,n] = out_val

    def __repr__(self):
        return "Peak shape clusterer"



def log_beta_pdf(x,a,b):
    o = gammaln(a + b) - gammaln(a) - gammaln(b)
    o = o + (a-1)*np.log(x) + (b-1)*np.log(1-x)
    return o


class shape_clusterer_greedy_sparse(object):
    def __init__(self,corr_mat,intensities,thresh=0.7,output = 1):
        self.thresh = thresh
        self.corr_mat = corr_mat
        self.intensities = intensities
        self.n_peaks = (self.corr_mat.shape)[0]
        self.output = output

    def _cluster(self):
        print "Running greedy clustering"
        self.Z = np.zeros(self.n_peaks,dtype=np.int64)-1
        temp_intensities = np.copy(self.intensities)
        finished = False
        cl_no = 0
        number_left = self.n_peaks
        self.counts = []
        while not finished:
            biggest_peak = np.argmax(temp_intensities)
            if self.output>0:
                print "Iteration " + str(cl_no) + ", " + str(number_left) + " peaks left"
            self.Z[biggest_peak] = cl_no
            temp_intensities[biggest_peak] = -1
            pos = np.where((self.corr_mat[biggest_peak,:]>=self.thresh)*(temp_intensities > 0))[0]
            self.Z[pos] = cl_no
            temp_intensities[pos] = -1
            number_left = np.sum(temp_intensities>0)
            self.counts.append((self.Z==cl_no).sum())
            cl_no += 1
            if number_left == 0:
                finished = True

    def _summary(self):
        print ""
        print "Greedy Clustering Summary"
        print "K = " + str(np.max(self.Z)+1)
        plt.figure()
        plt.pcolor(self.Z == self.Z[:,np.newaxis])
        plt.figure()
        plt.hist(np.sort(self.counts),np.arange(np.min(self.counts),np.max(self.counts)+1))
        plt.xlabel('Cluster size')
        plt.ylabel('Count')

class hyper(object):
    in_alpha = 10.0
    out_alpha = 1.0
    in_beta = 1.0
    out_beta = 10.0
    conc_par = 5.0
    in_prob = 0.99
    out_prob = 0.1

class data_generator_sparse(object):
    # This generates the data and stores in a sparse array (just the upper triangle)
    def __init__(self,hyper,n_peaks = 100,seed = 123):
        self.hyper_pars = hyper
        self.n_peaks = n_peaks
        self.seed = seed
        np.random.seed(self.seed)

    def _make_data(self):
        self.Z = []
        self.counts = []
        self.Z.append(1)
        self.counts.append(1)
        for n in np.arange(1,self.n_peaks):
        #     List concatanation
            temp = np.array(self.counts + [self.hyper_pars.conc_par])
            temp = temp/temp.sum()
            prob = temp.cumsum()
            pos = np.nonzero(np.random.rand()<prob)[0][0]
            self.Z.append(pos)
            if pos >= len(self.counts):
                self.counts.append(1)
            else:
                self.counts[pos] += 1

        self.Z = np.sort(self.Z)
        self.K = np.max(self.Z)
        self.intensities = []
        for n in np.arange(self.n_peaks):
            self.intensities.append(np.random.rand())
        # self.corr_mat = np.zeros((self.n_peaks,self.n_peaks))
        self.corr_mat = lil_matrix((self.n_peaks,self.n_peaks),dtype=np.float64)
        for n in np.arange(self.n_peaks-1):
            this_cluster = self.Z[n]
            for m in np.arange(n+1,self.n_peaks):
                if self.Z[m] == this_cluster:
                    if np.random.rand() < self.hyper_pars.in_prob:
                        this_val = beta.rvs(self.hyper_pars.in_alpha,self.hyper_pars.in_beta)
                    else:
                        this_val = 0
                else:
                    if np.random.rand() < self.hyper_pars.out_prob:
                        this_val = beta.rvs(self.hyper_pars.out_alpha,self.hyper_pars.out_beta)
                    else:
                        this_val = 0
                if this_val >0:
                    self.corr_mat[n,m] = this_val
                    self.corr_mat[m,n] = this_val
        # self.corr_mat = self.corr_mat.tocoo()


