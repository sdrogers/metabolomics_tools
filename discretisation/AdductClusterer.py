from collections import Counter
from random import shuffle
import time

from models import MassBin, IntervalTree
import numpy as np


class MassBinClusterer:
    
        def __init__(self, features, database, transformations, mass_tol, alpha, sigma, nsamps):
            ''' 
            Clusters peak features by the possible precursor masses, based on the specified list of adducts. 
            features is a list of peak features
            database is the list of database entries of molecules, used for matching by masses to the bins
            transformations is a list of adduct transformations (e.g. M+H, M+2H) and their associated constants 
            '''
            print 'MassBinClusterer initialised'
            self.features = features
            self.database = database
            self.transformations = transformations

            self.alpha = alpha  # Dirichlet concentration smoothing parameter
            self.mass_tol = mass_tol  # the tolerance used for binning in bin_range()
            self.sigma = 1.0 / (sigma * sigma)  # precision for RT
            self.mu_zero = np.mean([f.rt for f in features])  # hyperparameter mean for RT
            self.tau_zero = 5E-3  # hyperparameter precision for RT
            
            self.nsamps = nsamps  # total number of samples
            
        def run(self):
            ''' Runs the actual clustering here '''
            bins = self.discretise(self.features, self.transformations, self.mass_tol)
            T = self.identify(bins, self.database)
            mapping = self.cluster(T, bins, self.transformations)
            return mapping
            
        def discretise(self, features, transformations, mass_tol):
            ''' 
            Splits the mass axis into discrete bins, centered at the precursor masses obtained 
            from computing the inverse M+H transformation for each observed peak
            '''
            masses = np.array([f.mass for f in features])
            rts = np.array([f.rt for f in features])
            proton = transformations[6].sub  # hardcoded for now!!
            precursor_masses = masses - proton
            lower, upper = self.bin_range(precursor_masses, mass_tol)
            the_bins = []
            bin_id = 1
            for i in np.arange(len(precursor_masses)):
                low = lower[i]
                up = upper[i]
                rt = rts[i]
                the_bins.append(MassBin(bin_id, low, up, rt))                        
                bin_id = bin_id + 1
            return the_bins
        
        def bin_range(self, center, tol):
            ''' Defines a bin range centered at center, plus minus some tolerance'''
            interval = center * tol * 1e-6
            upper = center + interval
            lower = center - interval
            return lower, upper
        
        def identify(self, bins, database):
            '''
            Identifies mass bin with molecules in the database
            '''
            T = IntervalTree(bins)
            for mol in database:
                matching_bins = T.search(mol.mass)
                for mbin in matching_bins:
                    mbin.add_molecule(mol)
            return T
        
        def cluster(self, T, bins, transformations):
            '''
            Performs Gibbs sampling to reassign peak to bins based on the possible 
            transformation and RTs
            '''

            # initially put all peak into one bin
            z = {}  # z_nk, tracks feature to bin assignment
            first_bin = bins[0]
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
                    
                    # enumerate all the possible transformations and the associated mass bins
                    matching_bins = []
                    precursor_masses = []
                    trans_type = []
                    for trans in self.transformations:
                        trans_mass = (f.mass - trans.sub) / trans.mul
                        mass_bins = T.search(trans_mass)  # can have more than 1 ?!
                        for mb in mass_bins:
                            precursor_masses.append(trans_mass)
                            trans_type.append(trans)
                            matching_bins.append(mb)
                            assert len(precursor_masses) == len(matching_bins)
                            assert len(trans_type) == len(matching_bins)
                    
                    # perform reassignment of peak to bin
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
                    feature_annotation[f] = trans_type[k].name + ' @ ' + str(precursor_masses[k])
                    
                cluster_sizes = [str(mb.get_features_count()) for mb in bins]
                c = Counter()
                c.update(cluster_sizes)
                time_taken = time.time() - start_time
                print('SAMPLE %3d\t%4.2fs\t%s' % ((s+1), time_taken, str(c)))
                        
            print 'DONE!'      
            print
            print 'Last sample report'
            count_empty_bins = 0
            bin_mols = []
            bin_mols_unique = set()
            # sort the bin by no. of features first (biggest first)
            bins.sort(key=lambda x: x.get_features_count(), reverse=True)
            # loop through bins and print stuff
            for mass_bin in bins:
                # count molecules annotated to bin
                bin_mols.extend(mass_bin.get_molecules())
                bin_mols_unique.update(mass_bin.get_molecules())
                if mass_bin.get_features_count() > 0:
                    # print header
                    print mass_bin
                    for mol in mass_bin.get_molecules():
                        print "\t" + str(mol)
                    table = []
                    table.append(['feature_id', 'mass', 'rt', 'intensity', 'annotation', 'gt_metabolite', 'gt_adduct'])
                    # print features in this mass bin
                    for f in mass_bin.get_features():
                        table.append([str(f.feature_id), str(f.mass), str(f.rt), str(f.intensity), 
                                      feature_annotation[f], str(f.gt_metabolite), str(f.gt_adduct)])
                    self.print_table(table)
                    print
                else:
                    count_empty_bins = count_empty_bins + 1
            print
            print 'Empty bins=' + str(count_empty_bins)
            print 'Occupied bins=' + str(len(bins) - count_empty_bins) 
            print 'Molecules annotated to bins=' + str(len(bin_mols))
            print 'Unique molecules annotated to bins=' + str(len(bin_mols_unique))
            
            return bins
        
        def print_table(self, table):
            col_width = [max(len(x) for x in col) for col in zip(*table)]
            for line in table:
                print "| " + " | ".join("{:{}}".format(x, col_width[i])
                                        for i, x in enumerate(line)) + " |"