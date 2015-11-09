import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import timeit
import gzip
import cPickle

from collections import namedtuple
import csv
from operator import attrgetter
from operator import itemgetter
import multiprocessing
from joblib import Parallel, delayed  
import time
from itertools import chain, combinations

from discretisation import utils
from discretisation.preprocessing import FileLoader
from ground_truth import GroundTruth
from matching import MaxWeightedMatching
from models import AlignmentFile, Feature, AlignmentRow
import numpy as np
import shared_bin_matching_plotter as plotter
from clustering_calls import _run_first_stage_clustering, _run_second_stage_clustering
from discretisation.mulsubs import transformation


AlignmentResults = namedtuple('AlignmentResults', ['peakset', 'prob'])

class SharedBinMatching:
    
    def __init__(self, input_dir, database_file, transformation_file, hyperpars, 
                 synthetic=False, limit_n=-1, gt_file=None, verbose=False, seed=-1, parallel=True):

        loader = FileLoader()
        self.hp = hyperpars
        print self.hp
        self.data_list = loader.load_model_input(input_dir, None, 0, 0, make_bins=False, synthetic=synthetic, 
                                                 limit_n=limit_n, verbose=verbose)

        sys.stdout.flush()
        self.file_list = loader.file_list
        self.input_dir = input_dir
        self.database_file = database_file
        self.transformation_file = transformation_file
        self.gt_file = gt_file
        self.verbose = verbose
        self.seed = seed

        if parallel:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = 1

        self.annotations = {}
        if transformation_file is not None and transformation_file != 'null':
            self.trans_list = transformation.load_from_file(transformation_file)
            self.trans_map, self.trans_idx = self._get_transformation_map(self.trans_list)  
            self.MH = None
            self.T = len(self.trans_list)
            for t in self.trans_list:
                if t.name=="M+H":
                    self.MH = t
                    break
        
    def run(self, match_mode, show_singleton=False):

        start_time = time.time()
        
        print "Match mode " + str(match_mode)
        if match_mode == 0: # matching based on the peak features alone

            matching_mass_tol = self.hp.across_file_mass_tol
            matching_rt_tol = self.hp.across_file_rt_tol
            file_data = self.data_list
            alignment_results = self._match_peak_features(file_data, matching_mass_tol, matching_rt_tol)
            
        elif match_mode == 1: # matching based on the MAP of precursor clustering

            # do first-stage precursor clustering        
            clustering_results = self._first_stage_clustering()        
            file_data = {}
            for j in range(len(clustering_results)): # for each file
    
                ac = clustering_results[j]
                file_clusters = ac.clusters
    
                # initialise some attributes during runtime -- because we don't want to change the object            
                for cluster in file_clusters:
                    cluster.members = []
                    cluster.origin = j                    
                    cluster.word_counts = np.zeros(self.T)

#                 # assign peaks above the threshold to this cluster
#                 for peak in ac.peaks:
#                     for poss in ac.possible[peak]:
#                         if poss.prob > self.hp.t:
#                             msg = "{:s}@{:3.5f}({:.4f})".format(poss.transformation.name, poss.cluster.mu_mass, poss.prob)            
#                             self._annotate(peak, msg)   
#                             poss.cluster.members.append((peak, poss))        

                # perform MAP assignment of peaks to their most likely cluster
                ac.map_assign()
                for peak in ac.peaks:
                    best_poss = ac.Z[peak]
                    msg = "{:s}@{:3.5f}({:.4f})".format(best_poss.transformation.name, best_poss.cluster.mu_mass, best_poss.prob)            
                    self._annotate(peak, msg)   
                    best_poss.cluster.members.append((peak, best_poss))        
    
                # keep track of the non-empty clusters                            
                print
                print "File %d clusters assignment " % j
                selected = []
                for cluster in file_clusters:
                    if len(cluster.members) > 0:
                        print "Cluster ID %d" % cluster.id
                        for peak, poss in cluster.members:                    
                            print "\tpeak_id %d mass %f rt %f intensity %f (%s %.3f)" % (peak.feature_id, peak.mass, peak.rt, peak.intensity, 
                                                               poss.transformation.name, poss.prob)
                            # update the binary flag for this transformation in the cluster
                            tidx = self.trans_idx[poss.transformation.name]
                            cluster.word_counts[tidx] = poss.prob
                        selected.append(cluster)
    
                file_data[j] = selected # set non-empty clusters to match within each file            

            matching_mass_tol = self.hp.across_file_mass_tol
            matching_rt_tol = self.hp.across_file_rt_tol
            alignment_results = self._match_precursor_bins(file_data, matching_mass_tol, matching_rt_tol)
            self.clustering_results = clustering_results
            self.file_data = file_data

        elif match_mode == 2: # match with DP clustering

            # do first-stage precursor clustering        
            clustering_results = self._first_stage_clustering()        
            all_nonempty_clusters = []
            for j in range(len(clustering_results)): # for each file
    
                ac = clustering_results[j]
                file_clusters = ac.clusters
    
                # initialise some attributes during runtime -- because we don't want to change the object            
                for cluster in file_clusters:
                    cluster.members = []
                    cluster.origin = j                    
                    cluster.word_counts = np.zeros(self.T)
                
                # assign peaks above the threshold to this cluster
                for peak in ac.peaks:
                    for poss in ac.possible[peak]:
                        if poss.prob > self.hp.t:
                            msg = "{:s}@{:3.5f}({:.4f})".format(poss.transformation.name, poss.cluster.mu_mass, poss.prob)            
                            self._annotate(peak, msg)   
                            poss.cluster.members.append((peak, poss))        
    
                # keep track of the non-empty clusters                            
                print
                print "File %d clusters assignment " % j
                selected = []
                for cluster in file_clusters:
                    if len(cluster.members) > 0:
                        print "Cluster ID %d" % cluster.id
                        for peak, poss in cluster.members:                    
                            print "\tpeak_id %d mass %f rt %f intensity %f (%s %.3f)" % (peak.feature_id, peak.mass, peak.rt, peak.intensity, 
                                                               poss.transformation.name, poss.prob)
                            # update the binary flag for this transformation in the cluster
                            tidx = self.trans_idx[poss.transformation.name]
                            cluster.word_counts[tidx] += 1
                        selected.append(cluster)
    
                all_nonempty_clusters.extend(selected) # collect across all files
            
            matching_results, samples_obtained = self._second_stage_clustering(all_nonempty_clusters)
            alignment_results = self._construct_alignment(matching_results, samples_obtained)
            self.clustering_results = clustering_results
            self.all_nonempty_clusters = all_nonempty_clusters
         
        # for any matching mode, we should get the same alignment results back
        self._print_report(alignment_results, show_singleton=show_singleton)
        self.alignment_results = alignment_results        

        print
        print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        print
        
        if self.gt_file is not None:           
            print "Evaluating performance"
            self._evaluate_performance()
        
    def save_project(self, project_out):
        start = timeit.default_timer()        
        self.last_saved_timestamp = str(time.strftime("%c"))
        with gzip.GzipFile(project_out, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            stop = timeit.default_timer()
            print "Project saved to " + project_out + " time taken = " + str(stop-start)

    @classmethod
    def resume_from(cls, project_in):
        start = timeit.default_timer()        
        with gzip.GzipFile(project_in, 'rb') as f:
            obj = cPickle.load(f)
            stop = timeit.default_timer()
            print "Project loaded from " + project_in + " time taken = " + str(stop-start)
            return obj
        
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
            try:
                for row in self.alignment_results:
                    features = row.peakset
                    prob = row.prob
                    for f in features:
                        try:
                            msg = self.annotations[f._get_key()]
                        except AttributeError:
                            msg = 'None'
                        except KeyError:
                            msg = 'None'
                        parent_filename = self.file_list[f.file_id]
                        peak_id = f.feature_id-1
                        mass = f.mass
                        rt = f.rt
                        out = [row_id, parent_filename, peak_id, mass, rt, prob, msg]
                        writer.writerow(out)        
                    row_id += 1
                print 'Output written to', output_path
            except AttributeError:
                print 'Nothing written to', output_path
        
        
    def _first_stage_clustering(self):
        
        # run precursor clustering for each file
        print "First stage clustering -- within_file_mass_tol=%.2f, within_file_rt_tol=%.2f, alpha=%.2f" % (self.hp.within_file_mass_tol, self.hp.within_file_rt_tol, self.hp.alpha_mass)
        sys.stdout.flush()
        clustering_results = Parallel(n_jobs=self.num_cores, verbose=10)(delayed(_run_first_stage_clustering)(
                                        j, self.data_list[j], self.hp, self.transformation_file) for j in range(len(self.data_list)))
        assert len(clustering_results) == len(self.data_list)        
        return clustering_results

    def _create_abstract_bins(self, cluster_list, mass_tol):
                    
        all_features = np.array(cluster_list)
                    
        # create equally-spaced bins from start to end
        precursor_mass_list = []
        for cl in cluster_list:
            pm = cl.mu_mass
            precursor_mass_list.append(pm) 
        precursor_masses = np.array(precursor_mass_list)
        min_val = np.min(precursor_masses)
        max_val = np.max(precursor_masses)
        
        # iteratively find the bin centres
        all_bins = []
        bin_start, bin_end = utils.mass_range(min_val, mass_tol)
        while bin_end < max_val:
            # find all clusters that fit here
            matching_idx = np.where((precursor_masses>=bin_start) & (precursor_masses<=bin_end))[0]            
            # store the current bin centre
            bin_centre = utils.mass_centre(bin_start, mass_tol)
            all_bins.append(bin_centre)
            # advance the bin
            bin_start, bin_end = utils.mass_range(bin_centre, mass_tol)
            bin_start = bin_end

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

        K = len(abstract_bins)
        N = len(all_features)        
        print "Total abstract bins=" + str(K) + " total features=" + str(N)
        return abstract_bins
    
    def _cp_summary(self, possible, cluster_membership):
        
        print "Cluster output"
        s = cluster_membership.sum(0)
        nnz = (s>0).sum()
        
        print "Number of non-empty clusters: " + str(nnz) + " (of " + str(s.size) + ")"
        si = (cluster_membership).sum(0)
        print
        
        print "Size: count"
        for i in np.arange(0,si.max()+1):
            print str(i) + ": " + str((si==i).sum())
        t = (possible.multiply(cluster_membership)).data
        t -= 1
        print
        
        print "Trans: count"
        for key in self.trans_map:
            print self.trans_map[key].name + ": " + str((t==key).sum())

    def _match_peak_features(self, file_data, mass_tol, rt_tol):

        print "Matching peak features"
        sys.stdout.flush()

        alignment_files = []
        for j in range(len(file_data)):

            features = file_data[j].features
            this_file = AlignmentFile("file_" + str(j), self.verbose)
            row_id = 0            
            for f in features:                
                # initialise feature into row
                alignment_row = AlignmentRow(row_id)
                alignment_row.features.append(f)
                row_id = row_id + 1
                this_file.rows.append(alignment_row)
            alignment_files.append(this_file)
            
        # do the matching
        Options = namedtuple('Options', 'dmz drt exact_match use_fingerprint verbose')
        my_options = Options(dmz = mass_tol, drt = rt_tol, 
                             exact_match = False, use_fingerprint = False,
                             verbose = self.verbose)
        matched_results = AlignmentFile("", True)
        num_files = len(alignment_files)        
        for i in range(num_files):
            alignment_file = alignment_files[i]
            matcher = MaxWeightedMatching(matched_results, alignment_file, my_options)
            matched_results = matcher.do_matching()      
            
        # produce alignment results
        alignment_results = []
        for row in matched_results.rows:
            res = AlignmentResults(peakset=row.features, prob=1.0)
            alignment_results.append(res)            
        return alignment_results
        
    def _match_precursor_bins(self, file_data, mass_tol, rt_tol):

        print "Matching precursor bins"
        sys.stdout.flush()

        alignment_files = []
        alignment_feature_to_precursor_cluster = {}        

        for j in range(len(self.data_list)):

            file_clusters = file_data[j]
            file_post_masses = [cluster.mu_mass for cluster in file_clusters]
            file_post_rts = [cluster.mu_rt for cluster in file_clusters]
            file_post_fingerprints = [cluster.word_counts for cluster in file_clusters]
            this_file = AlignmentFile("file_" + str(j), self.verbose)

            peak_id = 0
            row_id = 0            
            for n in range(len(file_clusters)):
                
                cluster = file_clusters[n]
                mass = file_post_masses[n]
                rt = file_post_rts[n]
                fingerprint = file_post_fingerprints[n]

                if j == 1 and cluster.id == 1326:
                    print "Stop here"

                # initialise alignment feature
                alignment_feature = Feature(peak_id, mass, 1, 1, rt, this_file, fingerprint=fingerprint)
                alignment_feature_to_precursor_cluster[alignment_feature] = cluster

                # initialise row
                alignment_row = AlignmentRow(row_id)
                alignment_row.features.append(alignment_feature)

                peak_id = peak_id + 1
                row_id = row_id + 1
                this_file.rows.append(alignment_row)

            alignment_files.append(this_file)
            
        # do the matching
        Options = namedtuple('Options', 'dmz drt exact_match use_fingerprint verbose')
        my_options = Options(dmz = mass_tol, drt = rt_tol, 
                             exact_match = False, use_fingerprint = False,
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
                cluster = alignment_feature_to_precursor_cluster[alignment_feature]
                if cluster.id == 1326:
                    print "FOUND"
                    print "Cluster %d %.4f %.2f" % (cluster.id, cluster.mu_mass, cluster.mu_rt)
                    for f, poss in cluster.members:
                        print "- id %s mass %.4f rt %.2f" % ((f._get_key(), f.mass, f.rt))                    
                temp.append(cluster)
            tup = tuple(temp)
            results.append(tup)
            
        # turn this into a matching of peak features
        alignment_results = []
        for bin_res in results:
            matched_list = self._match_adduct_features(bin_res)
            for features in matched_list:
                res = AlignmentResults(peakset=features, prob=1.0)
                alignment_results.append(res)
        
        return alignment_results

    def _second_stage_clustering(self, cluster_list):

        # sort the non-empty clusters by posterior mass        
        sorted_clusters = sorted(cluster_list, key = attrgetter('mu_mass'))

        # create abstract bins with the same across_file_mass_tol ppm
        abstract_bins = self._create_abstract_bins(sorted_clusters, self.hp.across_file_mass_tol)
        all_groups = abstract_bins.values() # each value is a group of precursors that have been grouped by mass

        print "Running second-stage clustering"
        sys.stdout.flush()   
        file_matchings = Parallel(n_jobs=self.num_cores, verbose=10)(delayed(_run_second_stage_clustering)(
                                    n, all_groups[n], self.hp, self.seed) for n in range(len(all_groups)))

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
                                            
            enumerate_powersets = False
            if enumerate_powersets:
                # might give a ridiculous amount of results
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
            else:
                # track the original set
                if bins not in counter:
                    counter[bins] = 1
                else:
                    counter[bins] += 1
                    
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
            matched_list = self._match_adduct_features(members) 
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
                try:
                    msg = self.annotations[f._get_key()]
                except AttributeError:
                    msg = 'None'
                except KeyError:
                    msg = 'None'
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
        tidx = {}
        t = 1
        for trans in transformations:
            tmap[t] = trans
            tidx[trans.name] = t-1
            t += 1
        return tmap, tidx
        
    def _match_adduct_features(self, group):
        results = []
        if len(group) == 1:
            # just singleton things
            for f in group[0].members:
                peak_feature = f[0]
                tup = (peak_feature, )
                results.append(tup)                        
        else:
            # match by the adduct types only
            results = self._simple_matching(group)
        return results

    def _simple_matching(self, group):

        # combine everything of the same type together
        adducts = {}
        for cluster in group:
            for peak, poss in cluster.members:
                trans_name = poss.transformation.name
                if trans_name in adducts:
                    adducts[trans_name].append(peak)
                else:
                    adducts[trans_name] = [peak]

        # all the values of the same type are now aligned
        results = []
        for trans_name in adducts:
            tup = tuple(adducts[trans_name])
            results.append(tup)
        return results
        
    def _evaluate_performance(self):
        gt = GroundTruth(self.gt_file, self.file_list, self.data_list)        
        gt.evaluate_probabilistic_alignment(self.alignment_results)