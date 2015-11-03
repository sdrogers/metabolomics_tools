import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from collections import namedtuple
import csv
import itertools
import math
from operator import attrgetter
from operator import itemgetter
import multiprocessing
from joblib import Parallel, delayed  
import time
from itertools import chain, combinations

from discretisation import utils
from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.models import HyperPars
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
from ground_truth import GroundTruth
from matching import MaxWeightedMatching
from models import AlignmentFile, Feature, AlignmentRow
import numpy as np
import pylab as plt
from second_stage_clusterer import DpMixtureGibbs
import shared_bin_matching_plotter as plotter
from clustering_calls import _run_first_stage_clustering, _run_second_stage_clustering


AlignmentResults = namedtuple('AlignmentResults', ['peakset', 'prob'])

class SharedBinMatching:
    
    def __init__(self, input_dir, database_file, transformation_file, hyperpars, 
                 synthetic=False, limit_n=-1, gt_file=None, verbose=False, seed=-1):

        loader = FileLoader()
        self.hp = hyperpars
        print self.hp
        self.data_list = loader.load_model_input(input_dir, database_file, transformation_file, 
                                                 self.hp.within_file_mass_tol, self.hp.within_file_rt_tol,
                                                 make_bins=False, synthetic=synthetic, 
                                                 limit_n=limit_n, verbose=verbose)

        sys.stdout.flush()
        self.file_list = loader.file_list
        self.input_dir = input_dir
        self.database_file = database_file
        self.transformation_file = transformation_file
        self.gt_file = gt_file
        self.verbose = verbose
        self.seed = seed
        self.num_cores = multiprocessing.cpu_count()
        self.annotations = {}
        
    def run(self, matching_mass_tol, matching_rt_tol, full_matching=False, show_singleton=False):

        start_time = time.time()

        all_bins, posterior_bin_rts, posterior_bin_masses, file_data = self._first_stage_clustering(full_matching)
        
        if full_matching:
            # match the precursor bins directly
            alignment_results = self._match_precursor_bins(file_data, matching_mass_tol, matching_rt_tol)
        else:
            # perform second stage clustering
            matching_results, samples_obtained = self._second_stage_clustering(all_bins, posterior_bin_rts)
            alignment_results = self._construct_alignment(matching_results, samples_obtained)
        
        self._print_report(alignment_results, show_singleton=show_singleton)
        self.alignment_results = alignment_results        

        print
        print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        print
        
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
                    msg = self.annotations[f._get_key()]
                    parent_filename = self.file_list[f.file_id]
                    peak_id = f.feature_id-1
                    mass = f.mass
                    rt = f.rt
                    out = [row_id, parent_filename, peak_id, mass, rt, prob, msg]
                    writer.writerow(out)        
                row_id += 1
        
        print 'Output written to', output_path
        
    def _first_stage_clustering(self, full_matching):
        
        # collect features from all files, sorted by mass ascending
        all_features = []
        for peak_data in self.data_list:
            all_features.extend(peak_data.features)    
        all_features = sorted(all_features, key = attrgetter('mass'))

        # create abstract bins with the same across_file_mass_tol ppm
        any_trans = self.data_list[0].transformations # since trans is the same across all files ...
        any_tmap = self._get_transformation_map(any_trans)            
        abstract_bins = self._create_abstract_bins(all_features, self.hp.across_file_mass_tol, any_trans)

        # discretise each file and run precursor clustering
        print "Discretising at within_file_mass_tol=" + str(self.hp.within_file_mass_tol) + \
                    " and across_file_mass_tol=" + str(self.hp.across_file_mass_tol)
        results = Parallel(n_jobs=self.num_cores, verbose=50)(delayed(_run_first_stage_clustering)(
                                        j, self.data_list[j], self.data_list[j].transformations, 
                                        abstract_bins, self.hp, full_matching) for j in range(len(self.data_list)))

        # process the results
        all_bins = []
        posterior_bin_rts = []            
        posterior_bin_masses = []            
        file_data = {}
        for j in range(len(self.data_list)):
                                
            print "Processing first-stage clustering results for file " + str(j)
            sys.stdout.flush()
            peak_data = self.data_list[j]
            res_tup = results[j]
            precursor_clustering = res_tup[0]
            peak_data.set_discrete_info(res_tup[1])
        
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
        
            # find the non-empty bins' posterior RT values
            bin_rts = precursor_clustering.cluster_rt_mean[nnz_idx]
            bin_masses = precursor_clustering.cluster_mass_mean[nnz_idx]
            # plotter.plot_bin_posterior_rt(bin_rts, j)
            bin_rts = bin_rts.ravel().tolist()
            bin_masses = bin_masses.ravel().tolist()
            posterior_bin_rts.extend(bin_rts)
            posterior_bin_masses.extend(bin_masses)

            file_bins = bins
            file_post_rts = bin_rts
            file_post_masses = bin_masses
            file_data[j] = (file_bins, file_post_masses, file_post_rts)            
                
            # assign peaks into their respective bins, 
            # this makes it easier when matching peaks across the same bins later
            # note: a peak can belong to multiple bins, depending on the choice of threshold t
            cx = precursor_clustering.Z.tocoo()
            for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                
                if cluster_membership[i, j] > 0:
                                    
                    f = peak_data.features[i]
                    bb = peak_data.bins[j] # copy of the common bin specific to file j
    
                    # annotate each feature by its precursor mass & adduct type probabilities, for reporting later
                    bin_prob = precursor_clustering.Z[i, j]
                    trans_idx = precursor_clustering.possible[i, j]
                    tran = any_tmap[trans_idx]
                    msg = "{:s}@{:3.5f}({:.4f})".format(tran.name, bb.mass, bin_prob)            
                    self._annotate(f, msg)   
                    bin_id = bb.bin_id
                    bin_origin = bb.origin
                    msg = "bin_{:d}_file{:d}".format(bin_id, bin_origin)                            
                    self._annotate(f, msg)                

                    # add feature, transformation and its probability into the concrete bin
                    item = (f, trans_idx, bin_prob)
                    bb.add_feature(item)    
    
                    # track the word counts too for each transformation
                    tidx = trans_idx-1  # we use trans_idx-1 because the value of trans goes from 1 .. T
                    bb.word_counts[tidx] += math.floor(bin_prob*100)

            peak_data.remove_discrete_info()            
            print

#             for bb in bins:
#                 wc = ""
#                 for c in bb.word_counts:
#                     wc += "%03d." % c
#                 print wc
                
        # plotter.plot_bin_vs_bin(file_bins, file_post_rts)
        return all_bins, posterior_bin_rts, posterior_bin_masses, file_data

    def _create_abstract_bins(self, all_features, mass_tol, transformations):
                    
        all_features = np.array(all_features) # convert list to np array for easy indexing

        adduct_name = np.array([t.name for t in transformations])[:,None]      # A x 1
        adduct_mul = np.array([t.mul for t in transformations])[:,None]        # A x 1
        adduct_sub = np.array([t.sub for t in transformations])[:,None]        # A x 1
        proton_pos = np.flatnonzero(np.array(adduct_name)=='M+H')              # index of M+H adduct
                    
        # create equally-spaced bins from start to end
        feature_masses = np.array([f.mass for f in all_features])[:, None]              # N x 1
        precursor_masses = (feature_masses - adduct_sub[proton_pos])/adduct_mul[proton_pos]        
        min_val = np.min(precursor_masses)
        max_val = np.max(precursor_masses)
        
        # iteratively find the bin centres
        all_bins = []
        bin_start, bin_end = utils.mass_range(min_val, mass_tol)
        while bin_end < max_val:
            # store the current bin centre
            bin_centre = utils.mass_centre(bin_start, mass_tol)
            all_bins.append(bin_centre)
            # advance the bin
            bin_start, bin_end = utils.mass_range(bin_centre, mass_tol)
            bin_start = bin_end

        N = len(all_features)
        K = len(all_bins)
        T = len(transformations)
        print "Total abstract bins=" + str(K) + " total features=" + str(N) + " total transformations=" + str(T)

        print "Populating abstract bins ",
        abstract_bins = {}   
        k = 0
        for n in range(len(all_bins)):

            if n%10000==0:                        
                sys.stdout.write('.')
                sys.stdout.flush()            

            bin_centre = all_bins[n]
            interval_from, interval_to = utils.mass_range(bin_centre, mass_tol)
            matching_idx = np.where((precursor_masses>interval_from) & (precursor_masses<interval_to))[0]
            
            # if this abstract bin is not empty, then add features from all files that can fit here
            if len(matching_idx)>0:
                fs = all_features[matching_idx]
                data = fs.tolist()
                abstract_bins[k] = data
                k += 1        
        
        print        
        return abstract_bins

    def _match_precursor_bins(self, file_data, mass_tol, rt_tol):

        print "Matching precursor bins"
        sys.stdout.flush()

        alignment_files = []
        alignment_feature_to_bin = {}        

        for j in range(len(self.data_list)):
            
            file_id = j
            file_bins, file_post_masses, file_post_rts = file_data[j]
            this_file = AlignmentFile("file_" + str(file_id), self.verbose)

            peak_id = 0
            row_id = 0            
            for n in range(len(file_bins)):
                
                fbin = file_bins[n]
                mass = file_post_masses[n]
                rt = file_post_rts[n]

                # initialise alignment feature
                alignment_feature = Feature(peak_id, mass, 1, 1, rt, this_file)
                alignment_feature_to_bin[alignment_feature] = fbin

                # initialise row
                alignment_row = AlignmentRow(row_id)
                alignment_row.features.append(alignment_feature)

                peak_id = peak_id + 1
                row_id = row_id + 1
                this_file.rows.append(alignment_row)

            alignment_files.append(this_file)
            
        # do the matching
        Options = namedtuple('Options', 'dmz drt alignment_method exact_match use_group use_peakshape absolute_mass_tolerance mcs grouping_method alpha grt dp_alpha num_samples burn_in skip_matching always_recluster verbose')
        my_options = Options(dmz = mass_tol, drt = rt_tol, alignment_method = 'mw', exact_match = False, 
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
            
        # map the results back to the original bin objects
        results = []
        for row in matched_results.rows:
            temp = []
            for alignment_feature in row.features:
                fbin = alignment_feature_to_bin[alignment_feature]
                temp.append(fbin)
            tup = tuple(temp)
            results.append(tup)
            
        # turn this into a matching of peak features
        alignment_results = []
        for bin_res in results:
            matched_list = self._match_features(bin_res)
            for features in matched_list:
                res = AlignmentResults(peakset=features, prob=1.0)
                alignment_results.append(res)
        
        return alignment_results

    def _second_stage_clustering(self, all_bins, posterior_bin_rts):

        # Second-stage clustering
        N = len(all_bins)
        assert N == len(posterior_bin_rts)

        print "Selecting non-empty concrete bins for second-stage clustering"     
        sys.stdout.flush()   
        # get all the unique top ids of the non-empty concrete bins
        top_ids = [bb.top_id for bb in all_bins]
        top_ids = list(set(top_ids))

        # gather all the information we need for second-stage clustering for each abstract bin
        abstract_data = {}
        for n in range(len(top_ids)):
            
            selected_rts = []
            selected_word_counts = []
            selected_origins = []
            selected_bins = []
            
            # find all concrete bins under this top id
            if self.verbose:
                print
            for b in range(len(all_bins)):
                bb = all_bins[b]
                rt = posterior_bin_rts[b]
                if bb.top_id == top_ids[n]:
                    if self.verbose:
                        print " - " + str(bb) + " posterior RT = " + str(rt)
                    selected_rts.append(rt)
                    selected_word_counts.append(bb.word_counts)
                    selected_origins.append(bb.origin)
                    selected_bins.append(bb)

            abstract_data[n] = (selected_rts, selected_word_counts, selected_origins, selected_bins)

        print "Running second-stage clustering"
        sys.stdout.flush()   
        file_matchings = Parallel(n_jobs=self.num_cores, verbose=50)(delayed(_run_second_stage_clustering)(
                                    n, top_ids[n], len(top_ids), abstract_data[n], self.hp, self.seed
                                    ) for n in range(len(top_ids)))

        matching_results = []
        for file_res in file_matchings:
            matching_results.extend(file_res)
        samples_obtained = self.hp.rt_clustering_nsamps - self.hp.rt_clustering_burnin
                    
        return matching_results, samples_obtained

    def _construct_alignment(self, matching_results, samples_obtained, show_plot=False):
        
        print "Constructing alignment of peak features"
        
        # count frequencies of aligned bins produced across the Gibbs samples
        counter = dict()
        for n in range(len(matching_results)):
            
            bins = matching_results[n]
            if n%1000 == 0:
                print str(n) + "/" + str(len(matching_results))
                sys.stdout.flush()
                                                        
            # convert bins into its powerset, see http://stackoverflow.com/questions/18826571/python-powerset-of-a-given-set-with-generators
            for z in chain.from_iterable(combinations(bins, r) for r in range(len(bins)+1)):
                if len(z) == 0: # skip empty set
                    continue
                sorted_z = sorted(z, key = attrgetter('origin'))
                sorted_z = tuple(sorted_z)
                if sorted_z not in counter:
                    counter[sorted_z] = 1
                else:
                    counter[sorted_z] += 1
                    
        print
        
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
            matched_list = self._match_features(members) 
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
                msg = self.annotations[f._get_key()]            
                output = "\tfeature_id {:5d} file_id {:d} mz {:3.5f} RT {:5.2f} intensity {:.4e}\t{:s}".format(
                            f.feature_id, f.file_id, f.mass, f.rt, f.intensity, msg)
                print(output) 
            i += 1                         
                       
    def _annotate(self, feature, msg):
        if feature._get_key() in self.annotations:
            current_msg = self.annotations[feature._get_key()]
            self.annotations[feature._get_key()] = current_msg + ";" + msg
        else:
            self.annotations[feature._get_key()] = msg
        
    def _get_transformation_map(self, transformations):
        tmap = {}
        t = 1
        for trans in transformations:
            tmap[t] = trans
            t += 1
        return tmap
        
    def _match_features(self, members):
        results = []
        if len(members) == 1:
            # just singleton things
            for f in members[0].features:
                peak_feature = f[0]
                tup = (peak_feature, )
                results.append(tup)                        
        else:
            # match by the adduct types only
            results = self._simple_matching(members)
        return results

    def _simple_matching(self, members):
        
        # enumerate all the different adduct types
        adducts_list = [item[1] for bb in members for item in bb.features]
        adducts = set(adducts_list)
        
        # match peak features by each adduct type
        results = []
        for trans_to_collect in adducts:

            temp_res = []
            for bb in members: # within each concrete bin ..
                for item in bb.features:
                    peak_feature = item[0]
                    trans_idx = item[1]
                    trans_prob = item[2] # unused
                    # add feature with the right transition into the result
                    if trans_idx == trans_to_collect:
                        temp_res.append(peak_feature)
            tup = tuple(temp_res)
            results.append(tup)
                  
        return results
        
    def _evaluate_performance(self):
        gt = GroundTruth(self.gt_file, self.file_list, self.data_list)        
        gt.evaluate_probabilistic_alignment(self.alignment_results)