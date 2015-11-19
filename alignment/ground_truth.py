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
                    
                if len(gt_entry) > 0:
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
        
    def evaluate_alignment_results(self, peaksets, th_prob, annotations=None, feature_binning=None, verbose=False):   
                 
        tp = [] # should be matched and correctly matched
        fp = [] # should be matched but incorrectly matched
        fn = [] # should be matched but not matched
        isMHs = [] # is it matching by M+H adducts only?
        
        # compare against ground truth
        for i in range(len(self.gt_features)):

            group = self.gt_features[i]                        
            if verbose:
                print "Checking ground truth entry %d of length %d" % (i, len(group))
                for f in group:
                    key = f._get_key()
                    print "- id %s mass %.4f rt %.2f" % ((key, f.mass, f.rt))
        
            if verbose:
                print "Overlapping peaksets:"
                print
            overlap = self._find_overlap(group, peaksets)
            match = False
            isMH = False

            if len(overlap) == 0:            
                fp.append(i) # nothing matches, false positive

            elif len(overlap) == 1: # exactly one overlap

                ps, prob = overlap[0]
                match, isMH = self._print_peakset(ps, prob, group, annotations, feature_binning, verbose)
                if match: # and it's a correct match, so it's a true positive
                    tp.append(i)
                    isMHs.append(isMH)
                else: # else:
                    fp.append(i)
            
            else: # more than one overlaps
                fn.append(i)
            
        tp = float(len(tp))
        fp = float(len(fp))
        fn = float(len(fn))
            
        try:
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            f1 = (2*tp)/((2*tp)+fp+fn)
            return tp, fp, fn, prec, rec, f1, th_prob
        except ZeroDivisionError:
            return None        
                    
    def _find_overlap(self, gt_entry, aligned_peaksets):
        overlap = []
        for ps, prob in aligned_peaksets:
            ps_keys = [f._get_key() for f in ps]
            any_found = False
            for f in gt_entry:
                if f._get_key() in ps_keys:
                    any_found = True
            if any_found:
                overlap.append((ps, prob))
        return overlap    

    def _print_peakset(self, peakset, prob, gt_entry, annotations=None, feature_binning=None, verbose=True):
        if verbose:
            print "  Peakset %.2f" % prob
        features = list(peakset)
        features.sort(key=lambda x: x.file_id)    
        gt_keys = [g._get_key() for g in gt_entry]
        match = True
        isMH = False
        for f in features:
            key = f._get_key()
            if annotations is not None and key in annotations:
                if feature_binning is not None:
                    fbin = feature_binning[f._get_key()]
                    annot = annotations[key] + " top_bin " + str(fbin)
                else:
                    annot = annotations[key]
            else:
                annot = "None"            
            if 'M+H' in annot:
                isMH = True
            if verbose:
                print "  - id %s mass %.4f rt %.2f MAP_trans %s" % ((key, f.mass, f.rt, annot))    
            if key not in gt_keys:
                match = False
        if match:
            match_str = 'TRUE'
        else:
            match_str = 'FALSE'
        if verbose:
            print "  - Match=%s" % match_str
            print
        return match, isMH

    def _check_ground_truth(self, i, aligned_peaksets, aligner, feature_binning, verbose=True):
        
        group = self.gt_features[i]
        if verbose:
            print "Checking ground truth entry %d" % i
            for f in group:
                key = f._get_key()
                print "- id %s mass %.4f rt %.2f" % ((key, f.mass, f.rt))
    
        if verbose:
            print "Overlapping peaksets:"
            print
        overlap = self._find_overlap(group, aligned_peaksets)
        match = False
        isMH = False
        for ps, prob in overlap:
            match, isMH = self._print_peakset(ps, prob, group, aligner, feature_binning, verbose)
        return match, isMH
                
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