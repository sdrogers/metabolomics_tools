from collections import namedtuple
import csv
from itertools import chain, combinations
import multiprocessing
from operator import attrgetter
from operator import itemgetter
import os
import sys
import time
from Queue import PriorityQueue

from joblib import Parallel, delayed  

from clustering_calls import _run_first_stage_clustering, _run_second_stage_clustering
from discretisation import utils
from discretisation.mulsubs import transformation
from ground_truth import GroundTruth
from matching import MaxWeightedMatching
from models import AlignmentFile, Feature, AlignmentRow
import numpy as np
import shared_bin_matching_plotter as plotter


sys.path.insert(1, os.path.join(sys.path[0], '..'))

AlignmentResults = namedtuple('AlignmentResults', ['peakset', 'prob'])

class SharedBinMatching:
    
    def __init__(self, data_list, database_file, transformation_file, hyperpars, 
                 synthetic=True, limit_n=-1, verbose=True, seed=1234567890, parallel=True, mh_biggest=True, use_vb=False):

        self.verbose = verbose
        self.data_list = data_list
        self.hp = hyperpars
        if self.verbose:
            print self.hp
        sys.stdout.flush()
        
        self.file_list = []
        for data in data_list:
            self.file_list.append(data.filename)
        self.database_file = database_file
        self.transformation_file = transformation_file
        self.seed = seed
        self.mh_biggest = mh_biggest
        self.use_vb = use_vb

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
        
    def run(self, match_mode, first_stage_clustering_results=None):

        start_time = time.time()
        
        if self.verbose:
            print "Match mode " + str(match_mode)
            
        if match_mode == 0: # matching based on the peak features alone

            matching_mass_tol = self.hp.across_file_mass_tol
            matching_rt_tol = self.hp.across_file_rt_tol
            file_data = self.data_list
            alignment_results = self._match_peak_features(file_data, matching_mass_tol, matching_rt_tol)
            
        elif match_mode == 1: # matching based on the MAP of precursor clustering

            # do first-stage precursor clustering if not present
            if first_stage_clustering_results is None:
                clustering_results = self._first_stage_clustering()        
            else:
                clustering_results = first_stage_clustering_results
                
            file_data = {}
            for j in range(len(clustering_results)): # for each file
    
                ac = clustering_results[j]
                file_clusters = ac.clusters
    
                # initialise some attributes during runtime -- because we don't want to change the object            
                for cluster in file_clusters:
                    cluster.members = []
                    cluster.origin = j                    
                    cluster.word_counts = np.zeros(self.T)

                # perform MAP assignment of peaks to their most likely cluster
                ac.map_assign()
                for peak in ac.peaks:
                    best_poss = ac.Z[peak]
                    msg = "mz={:.4f},rt={:.2f},precursor={:s}@{:3.5f}(rt={:.2f},members={:d},prob={:.4f})".format(
                                                                                peak.mass, peak.rt, 
                                                                                best_poss.transformation.name, 
                                                                                best_poss.cluster.mu_mass, 
                                                                                best_poss.cluster.mu_rt,
                                                                                best_poss.cluster.N,
                                                                                best_poss.prob)            
                    self._annotate(peak, msg)   
                    best_poss.cluster.members.append((peak, best_poss))        
    
                # keep track of the non-empty clusters                            
                if self.verbose:
                    print
                    print "File %d clusters assignment " % j
                selected = []
                for cluster in file_clusters:
                    if len(cluster.members) > 0:
                        if self.verbose:
                            print "Cluster ID %d" % cluster.id
                        for peak, poss in cluster.members:                    
                            if self.verbose:
                                print "\tpeak_id %d mass %f rt %f intensity %f (%s %.3f)" % (peak.feature_id, peak.mass, peak.rt, peak.intensity, 
                                                                   poss.transformation.name, poss.prob)
                        selected.append(cluster)    
                file_data[j] = selected # set non-empty clusters to match within each file            

            matching_mass_tol = self.hp.across_file_mass_tol
            matching_rt_tol = self.hp.across_file_rt_tol
            alignment_results = self._match_precursor_bins(file_data, matching_mass_tol, matching_rt_tol)
            self.clustering_results = clustering_results
            self.file_data = file_data

        elif match_mode == 2: # match with DP clustering

            # do first-stage precursor clustering if not present
            if first_stage_clustering_results is None:
                clustering_results = self._first_stage_clustering()        
            else:
                clustering_results = first_stage_clustering_results

            all_nonempty_clusters = []
            for j in range(len(clustering_results)): # for each file
    
                ac = clustering_results[j]
                file_clusters = ac.clusters
    
                # initialise some attributes during runtime -- because we don't want to change adduct clusterer code ...
                for cluster in file_clusters:
                    cluster.members = []
                    cluster.origin = j                    
                    cluster.word_counts = np.zeros(self.T)

                if self.hp.t > 0: # assign peaks above threshold to clusters
                    for peak in ac.peaks:
                        for poss in ac.possible[peak]:
                            if poss.prob > self.hp.t:
                                msg = "mz={:.4f},rt={:.2f},precursor={:s}@{:3.5f}({:.4f})".format(peak.mass, peak.rt, 
                                                                                              poss.transformation.name, 
                                                                                              poss.cluster.mu_mass, poss.prob)            
                                self._annotate(peak, msg)   
                                poss.cluster.members.append((peak, poss))    
                                poss.cluster.N += 1
                                poss.cluster.rt_sum += peak.rt
                                poss.cluster.mass_sum += poss.transformed_mass                            
                else: # perform MAP assignment of peaks to their most likely cluster
                    ac.map_assign()
                    for peak in ac.peaks:
                        best_poss = ac.Z[peak]
                        msg = "mz={:.4f},rt={:.2f},precursor={:s}@{:3.5f}({:.4f})".format(peak.mass, peak.rt, 
                                                                                      best_poss.transformation.name, 
                                                                                      best_poss.cluster.mu_mass, best_poss.prob)            
                        self._annotate(peak, msg)   
                        best_poss.cluster.members.append((peak, best_poss))        
    
                # keep track of the non-empty clusters                            
                if self.verbose:
                    print
                    print "File %d clusters assignment " % j
                selected = []
                for cluster in file_clusters:
                    if len(cluster.members) > 0:
                        if self.verbose:
                            print "Cluster ID %d" % cluster.id
                        for peak, poss in cluster.members:                    
                            if self.verbose:
                                print "\tpeak_id %d mass %f rt %f intensity %f (%s %.3f)" % (peak.feature_id, peak.mass, peak.rt, peak.intensity, 
                                                                   poss.transformation.name, poss.prob)
                            # update the counts for this transformation in the cluster
                            tidx = self.trans_idx[poss.transformation.name]
                            # use the binary count only
                            cluster.word_counts[tidx] += 1
                            # use some dodgy scaling of the probabilities
                            # cluster.word_counts[tidx] += round(poss.prob*100)
                        selected.append(cluster)
    
                all_nonempty_clusters.extend(selected) # collect across all files
            
            matching_results, samples_obtained, all_groups = self._second_stage_clustering(all_nonempty_clusters)
            alignment_results = self._construct_alignment(matching_results, samples_obtained)
            self.clustering_results = clustering_results
            self.all_nonempty_clusters = all_nonempty_clusters
            self.all_groups = all_groups
         
        # for any matching mode, we should get the same alignment results back
        if self.verbose:
            self._print_report(alignment_results, show_singleton=False)
        self.alignment_results = alignment_results        

        if self.verbose:
            print
            print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
            print
                        
    def save_output(self, output_path):
        
        if output_path is None:
            return        
        
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
                if self.verbose:
                    print 'Output written to', output_path
            except AttributeError:
                if self.verbose:
                    print 'Nothing written to', output_path

    def evaluate_performance(self, gt_file, verbose=False, print_TP=True, method=2):

        performance = []
        gt = GroundTruth(gt_file, self.file_list, self.data_list, verbose=verbose)
        probs = set([res.prob for res in self.alignment_results])
        if len(probs) == 1:        

            peaksets = [(res.peakset, res.prob) for res in self.alignment_results]
            if method == 1:
                results = gt.evaluate_alignment_results_1(peaksets, 1.0, annotations=self.annotations, feature_binning=None, verbose=verbose, print_TP=print_TP)    
            elif method ==  2:
                results = gt.evaluate_alignment_results_2(peaksets, 1.0, annotations=self.annotations, feature_binning=None, verbose=verbose, print_TP=print_TP)                    
            performance.append(results)
        
        else:

            sorted_probs = sorted(probs)
            for th_prob in sorted_probs:
                # print "Processing %.3f" % th_prob
                sys.stdout.flush()
                peaksets = []
                for ps in self.alignment_results:
                    if ps.prob > th_prob:
                        peaksets.append(ps)
                # print len(peaksets)
                if len(peaksets) > 0:
                    if method == 1:
                        results = gt.evaluate_alignment_results_1(peaksets, th_prob, annotations=self.annotations, feature_binning=None, verbose=verbose)  
                    elif method == 2:
                        results = gt.evaluate_alignment_results_2(peaksets, th_prob, annotations=self.annotations, feature_binning=None, verbose=verbose)                          
                    # print results
                    if results is not None:  
                        performance.append(results)

        return performance
            
    def _first_stage_clustering(self):
        
        # run precursor clustering for each file
        if self.verbose:
            print "First stage clustering -- within_file_mass_tol=%.2f, within_file_rt_tol=%.2f, alpha=%.2f" % (self.hp.within_file_mass_tol, self.hp.within_file_rt_tol, self.hp.alpha_mass)
        sys.stdout.flush()
        clustering_results = Parallel(n_jobs=self.num_cores, verbose=10)(delayed(_run_first_stage_clustering)(
                                        j, self.data_list[j], self.hp, self.transformation_file, self.mh_biggest, self.use_vb) for j in range(len(self.data_list)))
        assert len(clustering_results) == len(self.data_list)        
        return clustering_results

    def _create_abstract_bins(self, cluster_list, mass_tol):
                            
        q = PriorityQueue()
        for cl in cluster_list:
            pm = cl.mu_mass
            q.put([pm, cl])            

        groups = {}
        k = 0
        group = []
        while not q.empty():
            
            current_item = q.get()
            current_mass = current_item[0]
            current_cluster = current_item[1]
            group.append(current_cluster)

            # check if the next mass is outside tolerance            
            if len(q.queue) > 0:
                head = q.queue[0]
                head_mass = head[0]
                lower, upper = utils.mass_range(current_mass, mass_tol)                            
                if head_mass > upper:
                    # start a new group
                    groups[k] = group
                    group = []
                    k += 1
            else:
                # no more item
                groups[k] = group

        K = len(groups)
        N = len(cluster_list)        
        if self.verbose:
            print "Total abstract bins=" + str(K) + " total features=" + str(N)
        return groups
    
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

        if self.verbose:
            print "Matching peak features"
        sys.stdout.flush()

        alignment_files = []
        input_features_count = 0
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
                input_features_count += 1
            alignment_files.append(this_file)
            
        # do the matching
        Options = namedtuple('Options', 'dmz drt exact_match use_fingerprint verbose')
        my_options = Options(dmz = mass_tol, drt = rt_tol, 
                             exact_match = False, use_fingerprint = False,
                             verbose = self.verbose)
        matched_results = AlignmentFile("", True)
        num_files = len(alignment_files)        
        input_count = 0
        output_count = 0
        for i in range(num_files):
            if self.verbose:
                print "Processing file %d" % i
            alignment_file = alignment_files[i]
            input_count += len(alignment_file.get_all_features()) + len(matched_results.get_all_features())
            matched_results.reset_aligned_status()
            alignment_file.reset_aligned_status()
            matcher = MaxWeightedMatching(matched_results, alignment_file, my_options)
            matched_results = matcher.do_matching()      
            output_count += len(matched_results.get_all_features())
            assert input_count == output_count, "input %d output %d" % (input_count, output_count)
            
        # produce alignment results
        output_features_count = 0
        alignment_results = []
        for row in matched_results.rows:
            for f in row.features:
                output_features_count += 1
            res = AlignmentResults(peakset=row.features, prob=1.0)
            alignment_results.append(res)            
            
        assert input_features_count == output_features_count, "input %d output %d" % (input_features_count, output_features_count)
        return alignment_results
        
    def _match_precursor_bins(self, file_data, mass_tol, rt_tol):

        if self.verbose:
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
        input_count = 0
        output_count = 0
        for i in range(num_files):
            if self.verbose:
                print "Processing file %d" % i
            alignment_file = alignment_files[i]
            input_count += len(alignment_file.get_all_features()) + len(matched_results.get_all_features())
            matched_results.reset_aligned_status()
            alignment_file.reset_aligned_status()
            matcher = MaxWeightedMatching(matched_results, alignment_file, my_options)
            matched_results = matcher.do_matching()      
            output_count += len(matched_results.get_all_features())
            assert input_count == output_count, "input %d output %d" % (input_count, output_count)
            
        # map the results back to the original bin objects
        results = []
        for row in matched_results.rows:
            temp = []
            for alignment_feature in row.features:
                cluster = alignment_feature_to_precursor_cluster[alignment_feature]
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

        if self.verbose:
            print "Running second-stage clustering"
        sys.stdout.flush()   
        file_matchings = Parallel(n_jobs=self.num_cores, verbose=10)(delayed(_run_second_stage_clustering)(
                                    n, all_groups[n], self.hp, self.seed) for n in range(len(all_groups)))

        matching_results = []
        for file_res in file_matchings:
            matching_results.extend(file_res)
        samples_obtained = self.hp.rt_clustering_nsamps - self.hp.rt_clustering_burnin
                    
        return matching_results, samples_obtained, all_groups

    def _construct_alignment(self, matching_results, samples_obtained, show_plot=False):
        
        if self.verbose:
            print "Constructing alignment of peak features"
        
        # count frequencies of aligned bins produced across the Gibbs samples
        counter = dict()
        for n in range(len(matching_results)):
            
            bins = matching_results[n]
            if self.verbose and n%1000 == 0:
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
                    
        if self.verbose:
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
            if self.verbose and n % 1000 == 0:
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