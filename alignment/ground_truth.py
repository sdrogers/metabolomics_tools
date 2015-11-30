import glob
import os
import sys

from discretisation import utils
from discretisation.preprocessing import FileLoader
import numpy as np
import pylab as plt

class GroundTruth:
        
    def __init__(self, gt_file, file_list, file_peak_data, verbose=False):

        # map filename (without extension) to the file data
        self.file_data = {}
        assert len(file_list) == len(file_peak_data)
        for j in range(len(file_list)):
            base = os.path.basename(file_list[j])
            front_part = os.path.splitext(base)[0]
            self.file_data[front_part] = file_peak_data[j]
        
        # read ground truth
        self.gt_file = gt_file
        self.gt_lines = []
        self.gt_features = []

        size_map = {}
        old_format = True # ground truth can come in 2 different formats
        with open(gt_file, 'rb') as input_file:
            for line in input_file:
                tokens = line.split()
                if tokens[0] is '>':
                    old_format = False
                gt_entry = []
                if old_format:
                    # handle old lange-format ground truth
                    assert len(tokens) % 5 == 0
                    filenames = []
                    unknown = []
                    intensities = []
                    rts = []
                    masses = []
                    state = 0
                    for tok in tokens:
                        if state == 0:
                            base = os.path.basename(tok)
                            front_part = os.path.splitext(base)[0]                            
                            filenames.append(front_part)
                            state = 1 # go to next state
                        elif state == 1:
                            unknown.append(tok)
                            state = 2 # go to next state
                        elif state == 2:
                            intensities.append(float(tok))
                            state = 3 # go to next state
                        elif state == 3:
                            rts.append(float(tok))
                            state = 4 # go to next state
                        elif state == 4:
                            masses.append(float(tok))
                            state = 0 # cycle back to state 0
                    assert len(filenames) == len(intensities)
                    assert len(filenames) == len(rts)
                    assert len(filenames) == len(masses)
                    for g in range(len(filenames)):
                        if filenames[g] in self.file_data.keys():
                            f = self._find_features(filenames[g], masses[g], rts[g], intensities[g])
                            assert f is not None
                            gt_entry.append(f)                            
                else:
                    # handle better new ground truth format
                    print "Unsupported"
                    
                if len(gt_entry) > 1:
                    item = tuple(gt_entry)
                    self.gt_features.append(item)
                    item_length = len(item)
                    if item_length in size_map:
                        size_map[item_length] += 1
                    else:
                        size_map[item_length] = 1
                    
        print "Loaded " + str(len(self.gt_features)) + " ground truth entries"   
        print size_map
        print
                
    def evaluate_bins(self, file_bins, peak_feature_to_bin, results):

        # construct an all_bins x all_bins ground truth matrix
        all_bins = {}
        a = 0
        for bins in file_bins:
            for bb in bins:
                all_bins[bb] = a
                a += 1
        A = len(all_bins.keys())
        gt_mat = np.zeros((A, A))
        res_mat = np.zeros((A, A))

        # enumerate all the pairwise combinations for the ground truth
        for consensus in self.gt_features:
            for f1 in consensus:
                for f2 in consensus:
                    bins1 = peak_feature_to_bin[f1]
                    bins2 = peak_feature_to_bin[f2]
                    for b1 in bins1:
                        for b2 in bins2:
                            i = all_bins[b1]
                            j = all_bins[b2]
                            gt_mat[i, j] = 1
        plt.figure()
        plt.pcolor(gt_mat)
        plt.show()
        
        # turn results into a pairwise thing too
        for matched_bins in results:
            for bin1 in matched_bins:
                for bin2 in matched_bins:
                    if bin1 == bin2:
                        continue
                    else:
                        i = all_bins[bin1]
                        j = all_bins[bin2]
                        res_mat[i, j] = 1
        plt.figure()
        plt.pcolor(res_mat)
        plt.show()    
        
    def evaluate_alignment_results_1(self, alignment_results, th_prob, annotations=None, 
                                     feature_binning=None, verbose=False, print_TP=True):   
                 
        tp = set() # should be matched and correctly matched
        fp = set() # should be matched but incorrectly matched
        fn = set() # should be matched but not matched at all
        
        ground_truth = []
        for item in self.gt_features:
            feature_keys = frozenset([f._get_key() for f in item])
            ground_truth.append(feature_keys)    

        peaksets = []
        for item, prob in alignment_results:
            ps_keys = frozenset([f._get_key() for f in item])
            intersects = self._find_intersection(ps_keys, ground_truth)
            if len(intersects)>0: # only consider items that also appear in ground truth
                peaksets.append(ps_keys)    
                
        for i in range(len(ground_truth)):
            gt_item = ground_truth[i]
            if len(gt_item) == 1: # skip single entry ground truth?
                continue
            intersects = self._find_intersection(gt_item, peaksets)
            if len(intersects) == 1:
                # check the positives
                ps = intersects[0]
                same = (gt_item == ps)
                if same:
                    tp.add(gt_item)
                    if print_TP:
                        print "TP %d peakset = %s" % (i, self._get_annotated_string(ps, annotations))
                        print "TP %d groundtruth = %s" % (i, self._get_annotated_string(gt_item, annotations))
                        print "------------------------------------------------------------------------------------------"
                else:
                    print "FP %d peakset = %s" % (i, self._get_annotated_string(ps, annotations))
                    print "FP %d groundtruth = %s" % (i, self._get_annotated_string(gt_item, annotations))
                    print "------------------------------------------------------------------------------------------"
                    fp.add(gt_item)
            else:
                # check the negatives
                fn.add(gt_item)
                for ps in intersects:
                    print "FN %d peakset = %s" % (i, self._get_annotated_string(ps, annotations))
                print "FN %d groundtruth = %s" % (i, self._get_annotated_string(gt_item, annotations))
                print "------------------------------------------------------------------------------------------"
            
        tp_count = float(len(tp))
        fp_count = float(len(fp))
        fn_count = float(len(fn))                        
        try:
            prec = tp_count/(tp_count+fp_count)
            rec = tp_count/(tp_count+fn_count)
            f1 = (2*tp_count)/((2*tp_count)+fp_count+fn_count)
            return tp_count, fp_count, fn_count, prec, rec, f1, th_prob
        except ZeroDivisionError:
            return tp, fp, fn, 0, 0, 0, th_prob        

    def evaluate_alignment_results_2(self, alignment_results, th_prob, annotations=None, 
                                     feature_binning=None, verbose=False, print_TP=True):   
                 
        ground_truth = []
        all_ground_truth_features = set()
        for item in self.gt_features:
            feature_keys = [f._get_key() for f in item]
            ground_truth.append(feature_keys)    
            all_ground_truth_features.update(feature_keys)

        peaksets = []
        for item, prob in alignment_results:
            ps_keys = [f._get_key() for f in item]
            peaksets.append(ps_keys)

        g_plus = self._get_pairwise_peakset(ground_truth, whitelist=None)
        t = self._get_pairwise_peakset(peaksets, whitelist=all_ground_truth_features)
        
        # TP = should be aligned & are aligned = G+ intersect t
        tp = g_plus.intersection(t)
        
        # FN = should be aligned & aren't aligned = G+ \ t
        fn = g_plus - t

        # FP = shouldn't be aligned & are aligned = t \ G+
        fp = t - g_plus
                    
        tp_count = float(len(tp))
        fp_count = float(len(fp))
        fn_count = float(len(fn))                        
        try:
            prec = tp_count/(tp_count+fp_count)
            rec = tp_count/(tp_count+fn_count)
            f1 = (2*tp_count)/((2*tp_count)+fp_count+fn_count)
            return tp_count, fp_count, fn_count, prec, rec, f1, th_prob
        except ZeroDivisionError:
            return tp, fp, fn, 0, 0, 0, th_prob        

    def _get_pairwise_peakset(self, peaksets, whitelist=None):
        
        results = []
        for ps in peaksets:
            if len(ps) == 1:
                results.append(ps)
            else:
                for item1 in ps:
                    for item2 in ps:
                        if item1 == item2:
                            continue
                        elif item1[1] > item2[1]:
                            continue
                        else:
                            results.append((item1, item2))

        if whitelist is not None:            
            unique_res = []                            
            for items in results:
                if len(items) == 1:
                    if items[0] not in whitelist:
                        continue
                elif len(items) == 2:
                    item1 = items[0]
                    item2 = items[1]
                    if item1 not in whitelist and item2 not in whitelist:
                        continue
                    else:
                        unique_res.append((item1, item2))
            return set(unique_res)
        else:
            return set(results)            

    def _get_annotated_string(self, peakset, annotations):
        output = "\n"
        for item in peakset:
            if item in annotations:
                output += "%s(%s)\n" % (item, annotations[item])
            else:
                output += str(item) + " "
        return output

    def _find_intersection(self, item, peakset):
        intersects = []
        for ps in peakset:
            same_elements = ps & item
            if len(same_elements) > 0:
                intersects.append(ps)
        return intersects
                    
    def _find_features(self, filename, mass, rt, intensity):
        EPSILON = 0.0001;
        features = self.file_data[filename].features
        for f in features:
            if abs(f.mass - mass) < EPSILON and abs(f.rt - rt) < EPSILON and abs(f.intensity - intensity) < EPSILON:
                return f
                
def main(argv):    
                
    database_file = '/home/joewandy/git/metabolomics_tools/discretisation/database/std1_mols.csv'
    transformation_file = '/home/joewandy/git/metabolomics_tools/discretisation/mulsubs/mulsub2.txt'
    input_dir = './input/std1_csv_2'
    gt_file = '/home/joewandy/git/metabolomics_tools/alignment/input/std1_csv_2/ground_truth/std1.positive.dat'
    gt = GroundTruth(gt_file, database_file, transformation_file, input_dir)
    
if __name__ == "__main__":
   main(sys.argv[1:])