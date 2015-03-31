
import numpy as np
from scipy.special import gammaln

class shape_cluster_gibbs(object):

    def __init__(self,corr_mat,hyper_pars,n_samples = 100,n_burn = 10,infinite = True,track = False,K=30,output = 0):
        self.corr_mat = corr_mat
        self.hyper_pars = hyper_pars

        self.n_samples = n_samples
        self.n_burn = n_burn

        self.infinite = infinite
        self.track = track
        self.K = 30

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
            Z = np.zeros(self.n_peaks,dtype=np.int64)
            counts = np.zeros(self.K).tolist()
            counts[0] = self.n_peaks

            if track:
                self.temp = np.tile(out_like.sum(axis=1),(self.K,1))
                self.temp[0,:] = in_like.sum(axis=1)

    def _sample(self):
        for samp in np.arange(self.n_samples):
            if self.output>0 and samp%10==0:
                print "Sample " + str(samp)
                for peak in np.arange(self.n_peaks):
                    this_peak = peak
                    this_cluster = self.Z[this_peak]
                    self.Z[this_peak] = -1
                    self.counts[this_cluster]-=1  
                    if self.track:
                        self.temp -= self.out_like[this_peak,:]
                        self.temp[this_cluster,:]+=self.out_like[this_peak,:]-self.in_like[this_peak,:]     



                        

    def _create_like_mats(self):
        print "Creating likelihood matrices"
        self.in_like = np.zeros((self.n_peaks,self.n_peaks))
        self.out_like = np.zeros((self.n_peaks,self.n_peaks))
        for n in np.arange(self.n_peaks-1):
            for m in np.arange(n+1,self.n_peaks):
                if self.corr_mat[n,m]!=0:
                    in_val = np.log(self.hyper_pars.in_prob) + log_beta_pdf(self.corr_mat[n,m],self.hyper_pars.in_alpha,self.hyper_pars.in_beta)
                    out_val = np.log(self.hyper_pars.out_prob) + log_beta_pdf(self.corr_mat[n,m],self.hyper_pars.out_alpha,self.hyper_pars.out_beta)
                else:
                    in_val = np.log(1-self.hyper_pars.in_prob)
                    out_val = np.log(1-self.hyper_pars.out_prob)
                
                self.in_like[n][m] = in_val
                self.in_like[m][n] = in_val
                self.out_like[n][m] = out_val
                self.out_like[m][n] = out_val



    def __repr__(self):
        return "Peak shape clusterer"



def log_beta_pdf(x,a,b):
    o = gammaln(a + b) - gammaln(a) - gammaln(b)
    o = o + (a-1)*np.log(x) + (b-1)*np.log(1-x)
    return o

class hyper(object):
    in_alpha = 10.0
    out_alpha = 1.0
    in_beta = 1.0
    out_beta = 10.0
    conc_par = 5.0
    in_prob = 0.99
    out_prob = 0.1

