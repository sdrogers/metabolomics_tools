from random import shuffle
import time

from scipy.special import psi

import numpy as np
import plotting
import scipy.sparse as sp


class DiscreteGibbs:
    
    def __init__(self, peak_data, hyperpars):
        ''' 
        Clusters peak features by the possible precursor masses, based on the specified list of adducts. 
        '''
        print 'DiscreteGibbs initialised'
        self.features = peak_data.features
        self.transformations = {}
        for t in peak_data.transformations:
            self.transformations[t.trans_id] = t # map id to transformation
        self.database = peak_data.database
        self.possible = peak_data.possible
        self.transformed = peak_data.transformed
        self.bins = peak_data.bins
        self.precursor_rts = peak_data.precursor_rts            
        self.n_samples = 20
        self.n_burn = 10
        self.alpha = float(hyperpars.alpha) # Dirichlet smoothing parameter
        self.sigma = float(hyperpars.rt_prec) # precision for RT
        self.tau_zero = float(hyperpars.rt_prior_prec) # hyperparameter precision for RT
        
        self.Z = sp.lil_matrix((peak_data.num_peaks, peak_data.num_peaks), dtype=np.float)
        self.cluster_rt_mean = np.zeros(peak_data.num_peaks)
        self.cluster_rt_prec = np.zeros(peak_data.num_peaks)
        
    def run(self):
        '''
        Performs Gibbs sampling to reassign peak to bins based on the possible 
        transformation and RTs
        '''

        # initially put all peak into its own bin
        Znk = {}  # z_nk, tracks feature to bin assignment
        feature_annotation = {}  # tracks f - transformation & precursor mass assignment
        for n in range(len(self.features)):
            f = self.features[n]
            current_bin = self.bins[n]
            current_bin.add_feature(f)
            Znk[f] = current_bin
            k = current_bin.bin_id
            adduct_id = self.possible[n, k]
            precursor = self.transformed[n, k]
            adduct = self.transformations[adduct_id]            
            feature_annotation[f] = adduct.name + ' @ ' + str(precursor)
            
        # now start the gibbs sampling  
        samples_taken = 0              
        for s in range(self.n_samples):

            # TODO: vectorise this part onwards!
            start_time = time.time()

            # for f in self.features:                
            random_order = range(len(self.features))
            shuffle(random_order)
            for n in random_order:

                f = self.features[n]
                
                # find possible target clusters
                current_bin = Znk[f]
                possible_clusters = np.nonzero(self.possible[n, :])[1]
                if len(possible_clusters) == 1:
                    # no reassignment to be done if only 1 possible cluster
                    if s >= self.n_burn:
                        self.Z[n, current_bin.bin_id] += 1.0                    
                    continue

                # remove peak from model
                current_bin.remove_feature(f)
                                                                                                
                # perform reassignment of peak to bin
                idx = list(possible_clusters)
                matching_bins = [self.bins[k] for k in idx]
                possible_adducts_ids = [self.possible[n, k] for k in idx]
                possible_precursor_masses = [self.transformed[n, k] for k in idx]
                possible_precursor_rts = [self.precursor_rts[n, k] for k in idx]

                log_posts = []
                for k in range(len(matching_bins)):

                    mass_bin = matching_bins[k]
                    mass_bin_rt = possible_precursor_rts[k]

                    # compute prior
                    log_prior = np.log(self.alpha + mass_bin.get_features_count())
                    
                    # compute mass likelihood
                    log_likelihood_mass = np.log(1.0) - np.log(len(matching_bins))
                                            
                    # compute RT likelihood -- mu is a random variable, marginalise this out
                    param_beta = self.tau_zero + (self.sigma * mass_bin.get_features_count())
                    temp = (self.tau_zero * mass_bin_rt) + (self.sigma * mass_bin.get_features_rt())
                    mu = (1 / param_beta) * temp
                    prec = 1 / ((1 / param_beta) + (1 / self.sigma))

                    # for plotting
                    self.cluster_rt_mean[n] = mu
                    self.cluster_rt_prec[n] = prec
                    
                    log_likelihood = -0.5 * np.log(2 * np.pi)
                    log_likelihood = log_likelihood + 0.5 * np.log(prec)
                    log_likelihood_rt = log_likelihood - 0.5 * np.multiply(prec, np.square(f.rt - mu))
                    
                    # compute posterior
                    log_post = log_prior + log_likelihood_mass + log_likelihood_rt
                    log_posts.append(log_post)

                # sample for the bin with the largest posterior probability
                assert len(log_posts) == len(matching_bins)
                log_posts = np.array(log_posts)
                log_posts = np.exp(log_posts - log_posts.max())
                log_posts = log_posts / log_posts.sum()
                random_number = np.random.rand()
                cumsum = np.cumsum(log_posts)
                k = 0
                for k in range(len(cumsum)):
                    c = cumsum[k]
                    if random_number <= c:
                        break
                found_bin = matching_bins[k]
                found_bin.add_feature(f)
                Znk[f] = found_bin
                adduct_id = possible_adducts_ids[k]
                precursor = possible_precursor_masses[k]
                adduct = self.transformations[adduct_id]
                feature_annotation[f] = adduct.name + ' @ ' + str(precursor)
                if s >= self.n_burn:
                    self.Z[n, found_bin.bin_id] += 1.0

            # done looping through all todo features
            time_taken = time.time() - start_time
            if s >= self.n_burn:
                # store sample
                samples_taken += 1
                plotting.print_cluster_sizes(self.bins, s, time_taken, True)
            else:
                # discard sample
                plotting.print_cluster_sizes(self.bins, s, time_taken, False)
                    
        print 'DONE!'      
        print
        plotting.print_last_sample(self.bins, feature_annotation)
        self.Z /= samples_taken
                
    def __repr__(self):
        return "Gibbs sampling for discrete mass model\n" + self.hyperpars.__repr__() + \
        "\nn_samples = " + str(self.n_samples)

class DiscreteVB:
    
    def __init__(self, peak_data, hyperpars):
        ''' 
        Clusters peak features by the possible precursor masses, 
        based on the specified list of adducts, using variational Bayes.
        '''
        print 'DiscreteVB initialised'
        self.features = peak_data.features
        self.n_peaks = len(self.features)
        self.possible = peak_data.possible
        self.precursor_rts = peak_data.precursor_rts
        self.rt = np.copy(peak_data.rt)
        self.prior_rt = np.copy(peak_data.rt)

        self.n_iterations = 100
        self.hyperpars = hyperpars
        self.delta_0 = hyperpars.rt_prior_prec
        self.delta = hyperpars.rt_prec
        self.alpha = hyperpars.alpha
        
        # Initially assign all peaks to its own cluster
        self.Z = sp.identity(self.n_peaks, format="lil")
                    
    def run(self):

        # Find peaks with more than 1 possible clusters to reassign
        todo = np.nonzero((self.possible>0).sum(1)>1)[0]
        print str(todo.size) + " peaks to be re-sampled"
                    
        for it in range(self.n_iterations):
            print "Iteration " + str(it)
            
            count_Z = np.array(self.Z.sum(0))
            
            # update thetas
            alpha_ks = self.alpha/self.n_peaks
            dir_params = count_Z + alpha_ks
            E_theta = dir_params/dir_params.sum()
            E_log_theta = (psi(dir_params) - psi(dir_params.sum())).T              
            
            # update mus
            sum_Z = np.array(self.Z.multiply(self.precursor_rts).sum(0))
            b = self.delta_0 + self.delta*count_Z
            E_mu = (1.0/b) * (self.delta_0*self.prior_rt.T + self.delta*sum_Z)
            var_mu = (1.0/b)
            E_mu2 = var_mu + np.square(E_mu)
            
            # for plotting
            self.cluster_rt_mean = E_mu.T
            self.cluster_rt_prec = b.T
            
            # update Z
            oldQZ = sp.lil_matrix(self.Z,copy=True)    
            for i in range(todo.size):

                this_row = todo[0, i]
                this_pos = self.possible.getrowview(this_row).nonzero()[1]

                # this_RT = np.array(self.precursor_rts[this_row,this_pos].toarray())
                this_possible = np.array(self.possible.getrowview(this_row).data[0])[:, None].T
                this_RT = np.tile(self.rt[this_row],(1, this_possible.size))
                
                temp = E_log_theta[this_pos].T              
                temp += -0.5 * self.delta * np.square(this_RT)
                temp += self.delta * this_RT * E_mu[0, this_pos]
                temp += -0.5 * self.delta * E_mu2[0, this_pos]

                temp = np.exp(temp - temp.max())
                temp = temp/temp.sum()
                self.Z[this_row, this_pos] = temp

            QChange = ((oldQZ-self.Z).data**2).sum()
            print "Change in Z: " + str(QChange)
                
    def __repr__(self):
        return "Variational Bayes for discrete mass model\n" + self.hyperpars.__repr__() + \
        "\nn_iterations = " + str(self.n_iterations)