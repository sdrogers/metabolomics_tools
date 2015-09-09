import numpy as np

import sys
import itertools
from operator import attrgetter
from operator import itemgetter
import time

from discretisation.models import HyperPars as DiscretisationHyperPars
from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
from discretisation import utils as DiscretisationUtils

import shared_bin_matching_plotter as plotter
from models import HyperPars as AlignmentHyperPars
from dp_rt_clusterer import DpMixtureGibbs

class SharedBinMatching:
    
    def __init__(self, input_dir, database_file, transformation_file, hyperpars, synthetic=False, limit_n=-1):
        ''' 
        Clusters bins by DP mixture model using variational inference
        '''
        print 'DpMixtureVariational initialised'
        loader = FileLoader()
        self.hp = hyperpars
        self.data_list = loader.load_model_input(input_dir, database_file, transformation_file, 
                                                 self.hp.binning_mass_tol, self.hp.binning_rt_tol, 
                                                 synthetic=synthetic, limit_n=limit_n)
        self.input_dir = input_dir
        self.database_file = database_file
        self.transformation_file = transformation_file

        self.annotations = {}
        
    def run(self):

        print "Running first-stage clustering"
        all_bins, posterior_bin_rts = self.first_stage_clustering()
        
        print "Running second-stage clustering"
        matching_results, samples_obtained = self.second_stage_clustering(all_bins, posterior_bin_rts)

        print "Constructing alignment of peak features"
        self.construct_alignment(matching_results, samples_obtained)
        
    def first_stage_clustering(self):
        
        # First stage clustering. 
        # Here we cluster peak features by their precursor masses to the common bins shared across files.

        any_transformations = self.data_list[0].transformations
        tmap = self.get_transformation_map(any_transformations)
        all_bins = []
        posterior_bin_rts = []    
        
        file_bins = []
        file_post_rts = []
        
        for j in range(len(self.data_list)):
        
            # run precursor mass clustering
            peak_data = self.data_list[j]
            # plotter.plot_possible_hist(peak_data.possible, self.input_dir, self.hp.binning_mass_tol, self.hp.binning_rt_tol)
            
            print "Clustering file " + str(j) + " by precursor masses"
            discHp = DiscretisationHyperPars()
            discHp.rt_prec = 1.0/(self.hp.within_file_rt_sd*self.hp.within_file_rt_sd)
            discHp.alpha = self.hp.alpha_mass
            
            discrete = DiscreteVB(peak_data, discHp)
            # discrete = ContinuousVB(peak_data, hp)

            discrete.n_iterations = self.hp.mass_clustering_n_iterations
            print discrete
            discrete.run()
        
            # pick the non-empty bins for the second stage clustering
            cluster_membership = (discrete.Z>self.hp.t)
            s = cluster_membership.sum(0)
            nnz_idx = s.nonzero()[1]  
            nnz_idx = np.squeeze(np.asarray(nnz_idx)) # flatten the thing
        
            # find the non-empty bins
            bins = [peak_data.bins[a] for a in nnz_idx]
            all_bins.extend(bins)
            file_bins.append(bins)
        
            # find the non-empty bins' posterior RT values
            bin_rts = discrete.cluster_rt_mean[nnz_idx]
            # plotter.plot_bin_posterior_rt(bin_rts, j)
            bin_rts = bin_rts.ravel().tolist()
            posterior_bin_rts.extend(bin_rts)
            file_post_rts.append(bin_rts)
        
            # make some plots
            cp = ClusterPlotter(peak_data, discrete)
            cp.summary(file_idx=j)
            # cp.plot_biggest(3)        
        
            # assign peaks into their respective bins, 
            # this makes it easier when matching peaks across the same bins later
            # note: a peak can belong to multiple bins, depending on the choice of threshold t
            print "Annotating peaks by transformations"
            cx = cluster_membership.tocoo()
            for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                f = peak_data.features[i]
                bb = peak_data.bins[j] # copy of the common bin specific to file j
                bb.add_feature(f)    
                # annotate each feature by its precursor mass & adduct type probabilities, for reporting later
                bin_prob = discrete.Z[i, j]
                trans_idx = discrete.possible[i, j]
                tran = tmap[trans_idx]
                msg = "{:s}@{:3.5f} prob={:.2f}".format(tran.name, bb.mass, bin_prob)            
                self.annotate(f, msg)   
            print
                
        # plotter.plot_bin_vs_bin(file_bins, file_post_rts)
        return all_bins, posterior_bin_rts

    def second_stage_clustering(self, all_bins, posterior_bin_rts):

        # Second-stage clustering
        N = len(all_bins)
        assert N == len(posterior_bin_rts)
        
        # Here we cluster the 'concrete' common bins across files by their posterior RT values
        hp = DiscretisationHyperPars()
        hp.rt_prec = 1.0/(self.hp.across_file_rt_sd*self.hp.across_file_rt_sd)
        hp.rt_prior_prec = 5E-3
        hp.alpha = self.hp.alpha_rt
        data = (posterior_bin_rts, all_bins)
        dp = DpMixtureGibbs(data, hp, seed=1234567890)
        dp.nsamps = self.hp.rt_clustering_nsamps
        dp.burn_in = self.hp.rt_clustering_burnin
        dp.run() 

        plotter.plot_ZZ_all(dp.ZZ_all)
        return dp.matching_results, dp.samples_obtained

    def construct_alignment(self, matching_results, samples_obtained):
        
        # count frequencies of aligned bins produced across the Gibbs samples
        print "Counting frequencies of aligned peaksets"
        counter = dict()
        for bins in matching_results:
            if len(bins) > 1:
                bins = sorted(bins, key = attrgetter('origin'))
                bins = tuple(bins)
            if bins not in counter:
                counter[bins] = 1
            else:
                counter[bins] += 1
        
        # normalise the counts
        print "Normalising counts"
        S = samples_obtained
        for key, value in counter.items():
            new_value = float(value)/S
            counter[key] = new_value       
            
        # print report of aligned peaksets in descending order of probabilities
        print 
        print "=========================================================================="
        print "REPORT"
        print "=========================================================================="
        sorted_list = sorted(counter.items(), key=itemgetter(1), reverse=True)
        probs = []
        i = 0
        for item in sorted_list:
            members = item[0]
            if len(members)==1:
                continue # skip all the singleton stuff
            prob = item[1]
            matched_list = self.match_features(members)
            for features in matched_list:
                if len(features)==1:
                    continue
                mzs = np.array([f.mass for f in features])
                rts = np.array([f.rt for f in features])
                avg_mz = np.mean(mzs)
                avg_rt = np.mean(rts)
                print str(i+1) + ". avg m/z=" + str(avg_mz) + " avg RT=" + str(avg_rt) + " prob=" + str(prob)
                for f in features:
                    msg = self.annotations[f]            
                    output = "\tfeature_id {:5d} file_id {:d} mz {:3.5f} RT {:5.2f} intensity {:.4e}\t{:s}".format(
                                f.feature_id, f.file_id, f.mass, f.rt, f.intensity, msg)
                    print(output) 
                probs.append(prob)
                i += 1             
                
        probs = np.array(probs) 
        plotter.plot_aligned_peaksets_probabilities(probs)
                       
    def annotate(self, feature, msg):
        if feature in self.annotations:
            current_msg = self.annotations[feature]
            self.annotations[feature] = current_msg + " " + msg
        else:
            self.annotations[feature] = msg
        
    def get_transformation_map(self, transformations):
        tmap = {}
        t = 1
        for trans in transformations:
            tmap[t] = trans
            t += 1
        return tmap
        
    def match_features(self, members):
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
        return results

def main(argv):    

    start = time.time()

#     input_dir = './input/std1_csv_2'
#     database_file = '../discretisation/database/std1_mols.csv'
#     transformation_file = '../discretisation/mulsubs/mulsub2.txt'
#     alignment_hp = AlignmentHyperPars()    

    input_dir = './input/P1/100'
    database_file = '../discretisation/database/std1_mols.csv'
    transformation_file = '../discretisation/mulsubs/mulsub2.txt'
    alignment_hp = AlignmentHyperPars()    
    alignment_hp.binning_mass_tol = 300
    alignment_hp.binning_rt_tol = 5.0
    alignment_hp.within_file_rt_sd = 2.5
    alignment_hp.across_file_rt_sd = 60
    alignment_hp.alpha_mass = 100.0
    alignment_hp.alpha_rt = 100.0
    alignment_hp.t = 0.25
    alignment_hp.mass_clustering_n_iterations = 20
    alignment_hp.rt_clustering_nsamps = 20
    alignment_hp.rt_clustering_burnin = 10
        
    sb = SharedBinMatching(input_dir, database_file, transformation_file, 
                           alignment_hp, synthetic=True)
    sb.run()
    
    end = time.time()
    DiscretisationUtils.timer("TOTAL ELAPSED TIME", start, end)    
    
if __name__ == "__main__":
   main(sys.argv[1:])
