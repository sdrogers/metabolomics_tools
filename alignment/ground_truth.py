import glob
import os

from discretisation import utils
from discretisation.preprocessing import FileLoader
import numpy as np
import pylab as plt


class GroundTruth:
        
    def __init__(self, gt_file, file_list, file_peak_data):

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
                            filenames.append(tok)
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
                    self.gt_features.append(tuple(gt_entry))
                    
        print "Loaded " + str(len(self.gt_features)) + " ground truth entries"        
#         for gt_entry in self.gt_features:
#             print gt_entry
                
    def evaluate_bins(self, file_bins, feature_to_bin, bin_alignment):

        # construct an all_bins x all_bins ground truth matrix
        all_bins = {}
        a = 0
        for bins in file_bins:
            for bb in bins:
                all_bins[bb] = a
                a += 1
        A = len(all_bins.keys())
        gt_mat = np.zeros(A, A)
        
        for consensus in self.gt_features:
            # enumerate all the pairwise combinations
            for f1 in consensus:
                for f2 in consensus:
                    bin1 = feature_to_bin[f1]
                    bin2 = feature_to_bin[f2]
                    i = all_bins[bin1]
                    j = all_bins[bin2]
                    gt_mat[i, j] = 1
        
        plt.figure()
        plt.pcolor(gt_mat)
        plt.show()
                
    def _find_features(self, filename, mass, rt, intensity):
        EPSILON = 0.0001;
        features = self.file_data[filename].features
        for f in features:
            if abs(f.mass - mass) < EPSILON and abs(f.rt - rt) < EPSILON and abs(f.intensity - intensity) < EPSILON:
                return f


                
database = '/home/joewandy/git/metabolomics_tools/discretisation/database/std1_mols.csv'
transformation = '/home/joewandy/git/metabolomics_tools/discretisation/mulsubs/mulsub2.txt'
input_dir = './input/std1_csv_2'

# find all the .txt and csv files in input_dir
file_list = []
types = ('*.csv', '*.txt')
os.chdir(input_dir)
for files in types:
    file_list.extend(glob.glob(files))
file_list = utils.natural_sort(file_list)

file_peak_data = []
for j in range(len(file_list)):
    input_file = file_list[j]
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database, transformation, 0, 0, make_bins=False)
    for f in peak_data.features:
        f.file_id = j
    file_peak_data.append(peak_data)

gt_file = '/home/joewandy/git/metabolomics_tools/alignment/input/std1_csv_2/ground_truth/std1.positive.dat'
gt = GroundTruth(gt_file, file_list, file_peak_data)