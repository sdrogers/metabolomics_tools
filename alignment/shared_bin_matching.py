import numpy as np

import sys
import os
import itertools
from operator import attrgetter
from operator import itemgetter
from collections import namedtuple
import csv

import pylab as plt

sys.path.append('/home/joewandy/git/metabolomics_tools')
from discretisation.models import HyperPars
from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
from discretisation import utils
from ground_truth import GroundTruth
from models import AlignmentFile, Feature, AlignmentRow
from matching import MaxWeightedMatching

import shared_bin_matching_plotter as plotter
from dp_rt_clusterer import DpMixtureGibbs

AlignmentResults = namedtuple('AlignmentResults', ['peakset', 'prob'])

class SharedBinMatching:
    
    def __init__(self, input_dir, database_file, transformation_file, hyperpars, 
                 synthetic=False, limit_n=-1, gt_file=None, verbose=False, seed=-1):
        ''' 
        Clusters bins by DP mixture model using variational inference
        '''
        loader = FileLoader()
        self.hp = hyperpars
        self.data_list = loader.load_model_input(input_dir, database_file, transformation_file, 
                                                 self.hp.within_file_mass_tol, self.hp.within_file_rt_tol,
                                                 self.hp.across_file_mass_tol, synthetic=synthetic, 
                                                 limit_n=limit_n, verbose=verbose)
        sys.stdout.flush()
        self.file_list = loader.file_list
        self.input_dir = input_dir
        self.database_file = database_file
        self.transformation_file = transformation_file
        self.gt_file = gt_file
        self.verbose = verbose
        self.seed = seed

        self.annotations = {}
        
    def run(self, matching_mass_tol, matching_rt_tol, full_matching=False, show_singleton=False, show_plot=False):

        print "Running first-stage clustering"
        all_bins, posterior_bin_rts = self._first_stage_clustering()
        
        print "Running second-stage clustering"
        matching_results, samples_obtained = self._second_stage_clustering(all_bins, posterior_bin_rts)

        print "Constructing alignment of peak features"
        alignment_results = self._construct_alignment(matching_results, samples_obtained, 
                                                      matching_mass_tol, matching_rt_tol, 
                                                      full_matching=full_matching, show_plot=show_plot)
        self._print_report(alignment_results, show_singleton=show_singleton)
        self.alignment_results = alignment_results        
        
        if self.gt_file is not None:           
            print "Evaluating performance"
            self._evaluate_performance()
        
    def save_output(self, output_path):
        
        # if the directory doesn't exist, create it
        if not os.path.exists(os.path.dirname(output_path)):
            dir = os.path.dirname(output_path)
            if len(dir)>0:
                os.makedirs(dir)        
        
        # write the result out in sima format, separated by tab
        # the columns are:
        # 1. alignment group id, index starts from 0
        # 2. originating file name
        # 3. position of feature in data, index starts from 0
        # 4. mass of feature
        # 5. rt of feature
        # 6. probability of alignment group (additional)
        # 7. annotation (additional)
        with open(output_path, 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)            
            row_id = 0
            for row in self.alignment_results:
                features = row.peakset
                prob = row.prob
                for f in features:
                    msg = self.annotations[f]
                    parent_filename = self.file_list[f.file_id]
                    peak_id = f.feature_id-1
                    mass = f.mass
                    rt = f.rt
                    out = [row_id, parent_filename, peak_id, mass, rt, prob, msg]
                    writer.writerow(out)        
                row_id += 1
        
        print 'Output written to', output_path
        
    def _first_stage_clustering(self):
        
        # First stage clustering. 
        # Here we cluster peak features by their precursor masses to the common bins shared across files.

        any_transformations = self.data_list[0].transformations
        tmap = self._get_transformation_map(any_transformations)
        all_bins = []
        posterior_bin_rts = []    
        
        file_bins = []
        file_post_rts = []
        
        for j in range(len(self.data_list)):
        
            # run precursor mass clustering
            peak_data = self.data_list[j]
            # plotter.plot_possible_hist(peak_data.possible, self.input_dir, self.hp.within_file_mass_tol, self.hp.within_file_rt_tol)
            
            print "Clustering file " + str(j) + " by precursor masses"
            precursorHp = HyperPars()
            precursorHp.rt_prec = 1.0/(self.hp.within_file_rt_sd*self.hp.within_file_rt_sd)
            precursorHp.alpha = self.hp.alpha_mass
            
            precursor_clustering = DiscreteVB(peak_data, precursorHp)                        
            # precursorHp.mass_prec = 1.0/(self.hp.within_file_rt_sd*self.hp.within_file_rt_sd)
            # precursor_clustering = ContinuousVB(peak_data, precursorHp)

            precursor_clustering.n_iterations = self.hp.mass_clustering_n_iterations
            print precursor_clustering
            precursor_clustering.run()
        
            # make some plots
            cp = ClusterPlotter(peak_data, precursor_clustering, threshold=self.hp.t)
            cp.summary(file_idx=j)
            # cp.plot_biggest(3)        
        
            # pick the non-empty bins for the second stage clustering
            cluster_membership = (precursor_clustering.Z>self.hp.t)
            s = cluster_membership.sum(0)
            nnz_idx = s.nonzero()[1]  
            nnz_idx = np.squeeze(np.asarray(nnz_idx)) # flatten the thing
        
            # find the non-empty bins
            bins = [peak_data.bins[a] for a in nnz_idx]
            all_bins.extend(bins)
            file_bins.append(bins)
        
            # find the non-empty bins' posterior RT values
            bin_rts = precursor_clustering.cluster_rt_mean[nnz_idx]
            # plotter.plot_bin_posterior_rt(bin_rts, j)
            bin_rts = bin_rts.ravel().tolist()
            posterior_bin_rts.extend(bin_rts)
            file_post_rts.append(bin_rts)
                
            # assign peaks into their respective bins, 
            # this makes it easier when matching peaks across the same bins later
            # note: a peak can belong to multiple bins, depending on the choice of threshold t
            cx = precursor_clustering.Z.tocoo()
            for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                
                if cluster_membership[i, j] > 0:
                                    
                    f = peak_data.features[i]
                    bb = peak_data.bins[j] # copy of the common bin specific to file j
                    bb.add_feature(f)    
    
                    # annotate each feature by its precursor mass & adduct type probabilities, for reporting later
                    bin_prob = precursor_clustering.Z[i, j]
                    trans_idx = precursor_clustering.possible[i, j]
                    tran = tmap[trans_idx]
                    msg = "{:s}@{:3.5f}({:.4f})".format(tran.name, bb.mass, bin_prob)            
                    self._annotate(f, msg)   
                    bin_id = bb.bin_id
                    bin_origin = bb.origin
                    msg = "bin_{:d}_file{:d}".format(bin_id, bin_origin)                            
                    self._annotate(f, msg)                
    
                    # track the word counts too for each transformation
                    tidx = trans_idx-1  # we use trans_idx-1 because the value of trans goes from 1 .. T
                    bb.word_counts[tidx] += 1
            
            print

            for bb in bins:
                wc = ""
                for c in bb.word_counts:
                    wc += str(c)
                print wc
                
        # plotter.plot_bin_vs_bin(file_bins, file_post_rts)
        return all_bins, posterior_bin_rts

    def _second_stage_clustering(self, all_bins, posterior_bin_rts):

        # Second-stage clustering
        N = len(all_bins)
        assert N == len(posterior_bin_rts)

        hp = HyperPars()
        hp.rt_prec = 1.0/(self.hp.across_file_rt_sd*self.hp.across_file_rt_sd)
        hp.rt_prior_prec = 5E-6
        hp.alpha = self.hp.alpha_rt        
        hp.alpha = 1
        matching_results = []
        top_ids = [bb.top_id for bb in all_bins]
        top_ids = list(set(top_ids))
        for n in range(len(top_ids)):
            selected_bins = []
            selected_rts = []
            print "Processing top_id " + str(top_ids[n]) + "\t\t(" + str(n) + "/" + str(len(top_ids)) + ")",
            if self.verbose:
                print
            for b in range(len(all_bins)):
                bb = all_bins[b]
                rt = posterior_bin_rts[b]
                if bb.top_id == top_ids[n]:
                    if self.verbose:
                        print " - " + str(bb) + " posterior RT = " + str(rt)
                    selected_bins.append(bb)
                    selected_rts.append(rt)
            data = (selected_rts, selected_bins)
            dp = DpMixtureGibbs(data, hp, seed=self.seed)
            dp.nsamps = self.hp.rt_clustering_nsamps
            dp.burn_in = self.hp.rt_clustering_burnin
            dp.run() 
            matching_results.extend(dp.matching_results)
            print "\tlast_K = " + str(dp.last_K)
            
        # plotter.plot_ZZ_all(dp.ZZ_all)
        samples_obtained = self.hp.rt_clustering_nsamps - self.hp.rt_clustering_burnin
        return matching_results, samples_obtained

    def _construct_alignment(self, matching_results, samples_obtained, 
                             matching_mass_tol, matching_rt_tol, 
                             full_matching=True, show_plot=False):
        
        # count frequencies of aligned bins produced across the Gibbs samples
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
        S = samples_obtained
        normalised = dict()
        for key, value in counter.items():
            new_value = float(value)/S
            normalised[key] = new_value       
            
        # construct aligned peaksets in descending order of probabilities
        alignment_results = []
        sorted_list = sorted(normalised.items(), key=itemgetter(1), reverse=True)
        probs = []
        n = 0
        for item in sorted_list:
            members = item[0]
            prob = item[1]
            if n % 1000 == 0:
                print "Processing aligned bins " + str(n) + "/" + str(len(sorted_list)) + "\tprob " + str(prob)
                sys.stdout.flush()
            # members is a tuple of bins so the features inside need to be matched
            matched_list = self._match_features(members, full_matching, matching_mass_tol, matching_rt_tol) 
            for features in matched_list:
                res = AlignmentResults(peakset=features, prob=prob)
                alignment_results.append(res)
            n += 1
                
        if show_plot:
            probs = np.array(probs) 
            plotter.plot_aligned_peaksets_probabilities(probs)

        return alignment_results
    
    def _print_report(self, alignment_results, show_singleton=False):
        print 
        print "=========================================================================="
        print "REPORT"
        print "=========================================================================="
        i = 0
        for res in alignment_results:
            features = res.peakset
            prob = res.prob
            if len(features)==1 and not show_singleton:
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
            i += 1                         
                       
    def _annotate(self, feature, msg):
        if feature in self.annotations:
            current_msg = self.annotations[feature]
            self.annotations[feature] = current_msg + ";" + msg
        else:
            self.annotations[feature] = msg
        
    def _get_transformation_map(self, transformations):
        tmap = {}
        t = 1
        for trans in transformations:
            tmap[t] = trans
            t += 1
        return tmap
        
    def _match_features(self, members, full_matching, matching_mass_tol, matching_rt_tol):
        results = []
        if len(members) == 1:
            # just singleton things
            features = members[0].features
            for f in features:
                tup = (f, )
                results.append(tup)                        
        else:
            if full_matching:
                results = self._hungarian_matching(members, matching_mass_tol, matching_rt_tol)
            else:
                results = self._simple_matching(members, matching_mass_tol, matching_rt_tol)
        return results

    def _hungarian_matching(self, members, mass_tol, rt_tol):

        file_id = 0
        alignment_files = []
        alignment_feature_to_discrete_feature = {}

        # convert each bin in members into an alignment file        
        for bb in members:
            this_file = AlignmentFile("file_" + str(file_id), self.verbose)
            peak_id = 0
            row_id = 0
            for discrete_feature in bb.features:                
                
                # initialise alignment feature
                mass = discrete_feature.mass
                charge = 1
                intensity = discrete_feature.intensity
                rt = discrete_feature.rt
                alignment_feature = Feature(peak_id, mass, charge, intensity, rt, this_file)
                alignment_feature_to_discrete_feature[alignment_feature] = discrete_feature

                # initialise row
                alignment_row = AlignmentRow(row_id)
                alignment_row.features.append(alignment_feature)

                peak_id = peak_id + 1
                row_id = row_id + 1
                this_file.rows.append(alignment_row)
            
            file_id += 1
            alignment_files.append(this_file)

        # do the matching
        Options = namedtuple('Options', 'dmz drt alignment_method exact_match use_group use_peakshape absolute_mass_tolerance mcs grouping_method alpha grt dp_alpha num_samples burn_in skip_matching always_recluster verbose')
        my_options = Options(dmz = mass_tol, drt = rt_tol, alignment_method = 'mw', exact_match = True, 
                             use_group = False, use_peakshape = False, absolute_mass_tolerance=False, 
                             mcs = 0.9,                             # unused
                             grouping_method = 'posterior',         # unused
                             alpha = 0.5, grt = 2,                  # unused
                             dp_alpha = 1,                          # unused
                             num_samples = 100,                     # unused
                             burn_in = 100,                         # unused
                             skip_matching = False,                 # unused
                             always_recluster = True,               # unused
                             verbose = self.verbose)
        matched_results = AlignmentFile("", True)
        num_files = len(alignment_files)        
        for i in range(num_files):
            alignment_file = alignment_files[i]
            matcher = MaxWeightedMatching(matched_results, alignment_file, my_options)
            matched_results = matcher.do_matching()      
            
        # map the results back
        results = []
        for row in matched_results.rows:
            temp = []
            for alignment_feature in row.features:
                discrete_feature = alignment_feature_to_discrete_feature[alignment_feature]
                temp.append(discrete_feature)
            tup = tuple(temp)
            results.append(tup)
        return results

    def _simple_matching(self, members, mass_tol, rt_tol):
        results = []
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
                            # skip item that has been processed before
                            if f2 in processed:
                                continue
                            # check mass and rt tol
                            mass_ok = utils.mass_match(f1.mass, f2.mass, mass_tol)
                            rt_ok = utils.rt_match(f1.rt, f2.rt, rt_tol)
                            if not mass_ok or not rt_ok:
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
        
    def _evaluate_performance(self):
        gt = GroundTruth(self.gt_file, self.file_list, self.data_list)        
        gt.evaluate_probabilistic_alignment(self.alignment_results)