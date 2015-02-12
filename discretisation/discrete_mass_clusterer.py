from random import shuffle
import time

import sample_reporter

import numpy as np


class DiscreteGibbs:
    
        def __init__(self, peak_data, hyperpars, nsamps):
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

            # Dirichlet smoothing parameter
            self.alpha = hyperpars.discrete_alpha  

            # precision for RT
            self.sigma = 1.0 / (hyperpars.discrete_rt_stdev * hyperpars.discrete_rt_stdev)
            
            # hyperparameter mean for RT
            self.mu_zero = np.mean([f.rt for f in self.features])
            
            # hyperparameter precision for RT
            self.tau_zero = hyperpars.discrete_rt_prior_prec
            
            # total number of samples
            self.nsamps = nsamps  
            
        def run(self):
            '''
            Performs Gibbs sampling to reassign peak to bins based on the possible 
            transformation and RTs
            '''

            # initially put all peak into one bin
            z = {}  # z_nk, tracks feature to bin assignment
            first_bin = self.bins[0]
            for f in self.features:
                first_bin.add_feature(f)
                z[f] = first_bin
            feature_annotation = {}  # tracks f - transformation & precursor mass assignment
                
            # now start the gibbs sampling                
            for s in range(self.nsamps):

                start_time = time.time()

                # for f in self.features:                
                random_order = range(len(self.features))
                shuffle(random_order)
                for n in random_order:

                    f = self.features[n]

                    # remove peak from model
                    current_bin = z[f]
                    current_bin.remove_feature(f)
                    
                    # find possible target clusters
                    possible_clusters = np.nonzero(self.possible[n, :])[1]
                    if len(possible_clusters) == 1:
                        # no reassignment to be done
                        continue
                                                                                                    
                    # perform reassignment of peak to bin
                    idx = list(possible_clusters)
                    matching_bins = [self.bins[k] for k in idx]
                    possible_adducts_ids = [self.possible[n, k] for k in idx]
                    possible_precursor_masses = [self.transformed[n, k] for k in idx]

                    # TODO: we really should vectorise this bit!
                    log_posts = []
                    for mass_bin in matching_bins:

                        # compute prior
                        log_prior = np.log(self.alpha + mass_bin.get_features_count())
                        
                        # compute mass likelihood
                        log_likelihood_mass = np.log(1.0) - np.log(len(matching_bins))
                        
                        # compute RT likelihood -- mu is fixed to the RT of the peak that generates the M+H                        
                        # mu = mass_bin.get_rt()
                        # prec = self.sigma
                        
                        # compute RT likelihood -- mu is a random variable, marginalise this out
                        param_beta = self.tau_zero + (self.sigma * mass_bin.get_features_count())
                        temp = (self.tau_zero * self.mu_zero) + (self.sigma * mass_bin.get_features_rt())
                        mu = (1 / param_beta) * temp
                        prec = 1 / ((1 / param_beta) + (1 / self.sigma))
                        
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
                    z[f] = found_bin
                    adduct_id = possible_adducts_ids[k]
                    precursor = possible_precursor_masses[k]
                    adduct = self.transformations[adduct_id]
                    feature_annotation[f] = adduct.name + ' @ ' + str(precursor)

                time_taken = time.time() - start_time
                sample_reporter.print_cluster_sizes(self.bins, s, time_taken)
                        
            print 'DONE!'      
            print
            sample_reporter.print_last_sample(self.bins, feature_annotation)
            
            return self.bins
