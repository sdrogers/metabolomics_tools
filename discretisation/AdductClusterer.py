import os
import os.path
from random import shuffle
import sys
import time

import scipy.stats

from models import Feature, MassBin, IntervalTree
import numpy as np


class MassBinClusterer:
    
        def __init__(self, features, transformations, mass_tol, alpha, sigma, nsamps, burnin):
            ''' 
            Clusters peak features by the possible precursor masses, based on the specified list of adducts. 
            features is a list of peak features
            transformations is a list of adduct transformations (e.g. M+H, M+2H) and their associated constants 
            '''
            self.features = features
            self.transformations = transformations
            self.alpha = alpha # Dirichlet concentration parameter
            self.sigma = sigma # Normal standard deviation on the RT
            self.nsamps = nsamps # total number of samples
            self.burnin = burnin # burn-in samples to discard
            self.mass_tol = mass_tol
            
        def run(self):
            ''' Runs the actual clustering here '''
            bins = self.discretise(self.features, self.transformations, self.mass_tol)
            mapping = self.cluster(bins, self.transformations)
            return mapping
            
        def discretise(self, features, transformations, mass_tol):
            ''' 
            Splits the mass axis into discrete bins, centered at the precursor masses from doing the 
            inverse M+H transformation for each observed peak
            '''
            masses = np.array([f.mass for f in features])
            rts = np.array([f.rt for f in features])
            proton = transformations[6].sub # hardcoded for now!!
            precursor_masses = masses - proton
            lower, upper = self.bin_range(precursor_masses, mass_tol)
            the_bins = []
            bin_id = 1
            for i in np.arange(len(precursor_masses)):
                low = lower[i]
                up = upper[i]
                the_bins.append(MassBin(bin_id, low, up, rts[i]))                        
                bin_id = bin_id + 1
            return the_bins
        
        def bin_range(self, m1, tol):
            ''' Defines a bin range centered at m1, plus minus tolerance'''
            interval = m1*tol*1e-6
            upper = m1+interval
            lower = m1-interval
            return lower, upper
        
        def cluster(self, bins, transformations):
            '''
            Performs Gibbs sampling to reassign peak to bins
            for 1 .. nsamps
                loop through all peaks randomly
                    remove peak from model
                    reassign peak to bin
            '''

            # store all the bins in an interval tree for quick lookup by mass
            T = IntervalTree(bins)

            # put all bins into one cluster
            first_bin = bins[0]
            for feature in self.features:
                first_bin.add_feature(feature)

            # now start the gibbs sampling                
            for s in range(self.nsamps):
                print 's = ' + str(s)
                for feature in self.features:
                    # remove peak from model
                    for massbin in bins:
                        massbin.remove_feature(feature)
                    # enumerate all the possible transformations and the associated mass bins
                    unique_bins = set()
                    for trans in self.transformations:
                        trans_mass = (feature.mass - trans.sub) / trans.mul
                        mass_bins = T.search(trans_mass) # can have more than 1
                        # mass_bins = self.find_bins(bins, trans_mass)
                        for mb in mass_bins:
                            unique_bins.add(mb) # keep only unique ones
                    # reassign peak to bin
                    matching_bins = list(unique_bins)
                    posts = []
                    for mass_bin in matching_bins:
                        prior = self.alpha + mass_bin.get_features_count()
                        normal_dist = scipy.stats.norm(mass_bin.get_rt(), self.sigma) 
                        likelihood = normal_dist.pdf(feature.rt)
                        post = prior * likelihood # TODO: do this in log space
                        posts.append(post)
                    found_bin = None
                    if len(posts) == 1:
                        # if only one, then immediately add it in
                        found_bin = matching_bins[0]
                    else:
                        # sample from the posterior
                        posts = np.array(posts)
                        posts = posts / posts.sum()
                        random_number = np.random.rand()
                        cumsum = np.cumsum(posts)
                        k = 0
                        for k in range(len(cumsum)):
                            c = cumsum[k]
                            if random_number <= c:
                                break
                        found_bin = matching_bins[k]
                    found_bin.add_feature(feature)
                    print '\t' + str(feature) + ' goes into ' + str(found_bin)
                        
            print 'Sampling done!'
            for massbin in bins:
                print massbin
                
            return bins
        
        def find_bins(self, bins, mass):
            results = []
            for bin in bins:
                if bin.get_begin() < mass < bin.get_end():
                    results.append(bin)
            return results