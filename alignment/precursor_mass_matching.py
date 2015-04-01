from collections import namedtuple
import glob
import itertools
import os

from matching import MaxWeightedMatching
from discretisation import utils
from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.models import HyperPars
from discretisation.preprocessing import FileLoader
from models import AlignmentFile, Feature, AlignmentRow
import numpy as np


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

binning_mass_tol = 2.0                 # mass tolerance in ppm when binning
binning_rt_tol = 5.0                   # rt tolerance in seconds when binning
limit_n = 1000                         # the number of features to load per file to make debugging easier, -1 to load all
t = 0.5
    
# load alignment ground truth here


# First stage clustering. 
# Here we cluster peak features by their precursor masses to the common bins shared across files.
feature_to_bin = {}
file_bins = []
for j in range(len(file_list)):
    
    input_file = file_list[j]
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database, transformation, binning_mass_tol, binning_rt_tol, limit_n=limit_n)

    # run precursor mass clustering
    print "Clustering file " + input_file + " by precursor masses"
    hp = HyperPars()
    pc_clustering = ContinuousVB(peak_data, hp)
    pc_clustering.n_iterations = 20
    print pc_clustering
    pc_clustering.run()

    # pick the non-empty bins for the second stage clustering
    cluster_membership = (pc_clustering.Z>t)
    s = cluster_membership.sum(0)
    nnz_idx = s.nonzero()[1]  
    nnz_idx = np.squeeze(np.asarray(nnz_idx)) # flatten the thing

    # assign peaks into their respective bins, 
    # this makes it easier when matching peaks across the same bins later
    # note: a peak can belong to multiple bins, depending on the choice of threshold t
    cx = cluster_membership.tocoo()
    for row,col,val in itertools.izip(cx.row, cx.col, cx.data):
        f = peak_data.features[row]
        bb = peak_data.bins[col] # copy of the common bin specific to file j
        bb.add_feature(f)    

    # find the non-empty bins
    bins = [peak_data.bins[a] for a in nnz_idx]
    print "Non-empty bins=" + str(len(bins))

    # find the non-empty bins' posterior mass and RT values
    bin_masses = pc_clustering.cluster_mass_mean[nnz_idx]
    bin_rts = pc_clustering.cluster_rt_mean[nnz_idx]
    bin_masses = bin_masses.ravel().tolist()
    bin_rts = bin_rts.ravel().tolist()
    
    # initialise bin posterior mass and rt values, and also the avg intensity
    for a in range(len(bins)):
        bb = bins[a]
        bb.posterior_rt = bin_rts[a]
        bb.posterior_mass = bin_masses[a]
        intensities = np.array([f.intensity for f in bb.features])
        bb.avg_intensity = np.asscalar(np.mean(intensities))
        bb.origin = j
    file_bins.append(bins)

# make ground truth for bins

                    
# then match the bins across runs
assert len(file_list) == len(file_bins)
alignment_files = []
feature_to_bin = {}
for j in range(len(file_list)):
    bins = file_bins[j]
    print file_list[j] + " has " + str(len(bins)) + " bins"
    # convert bins into features
    this_file = AlignmentFile(file_list[j], True)
    row_id = 0
    peak_id = 0
    for bb in bins:
        # initialise feature
        mass = bb.posterior_mass
        charge = 1
        intensity = bb.avg_intensity
        rt = bb.posterior_rt
        feat = Feature(peak_id, mass, charge, intensity, rt, this_file)
        peak_id = peak_id + 1
        feature_to_bin[feat] = bb
        # initialise row
        alignment_row = AlignmentRow(row_id)
        alignment_row.features.append(feat)
        row_id = row_id + 1
        this_file.rows.append(alignment_row)
    # print summary
    row_count = str(len(this_file.rows))
    print " - " + row_count + " rows converted"
    alignment_files.append(this_file)

Options = namedtuple('Options', 'dmz drt alignment_method exact_match use_group use_peakshape mcs grouping_method alpha grt dp_alpha num_samples burn_in skip_matching always_recluster verbose')
my_options = Options(dmz = 0.01, drt = 30, alignment_method = 'mw', exact_match = True, 
                     use_group = False, use_peakshape = False, mcs = 0.9, 
                     grouping_method = 'posterior', alpha = 0.5, grt = 2, dp_alpha = 1, num_samples = 100, 
                     burn_in = 100, skip_matching = False, always_recluster = True,
                     verbose = True)
print my_options
matched_results = AlignmentFile("", True)
num_files = len(alignment_files)        
for i in range(num_files):
    alignment_file = alignment_files[i]
    # match the files
    print ("\nProcessing " + alignment_file.filename + " [" + str(i+1) + "/" + str(num_files) + "]")
    matcher = MaxWeightedMatching(matched_results, alignment_file, my_options)
    matched_results = matcher.do_matching()      
    
print "Matching results"
results = []
for row in matched_results.rows:
    matched_bins = []
    for f in row.features:
        bb = feature_to_bin[f]
        matched_bins.append(bb)
    matched_bins = tuple(matched_bins)
    results.append(matched_bins)
    print matched_bins
    
print "Performance evaluation"