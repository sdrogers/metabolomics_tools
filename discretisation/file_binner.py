import numpy as np
import scipy.sparse as sp
import sys

import utils
from models import DiscreteInfo, PrecursorBin

def _process_file(data_list, top_bins, top_bin_features, 
                  transformations, adduct_sub, adduct_mul, adduct_del, proton_pos, 
                  within_file_mass_tol, within_file_rt_tol, file_idx):

    peak_data = data_list[file_idx]
    features = peak_data.features
    N = len(features)     
    T = len(transformations)   

    # initialise the 'concrete' realisations of the top bins in this file
    concrete_bins = []
    k = 0
    for a in range(len(top_bins)):
                        
        # find all features that can fit by mass in the top level bin
        tb = top_bins[a]
        fs = top_bin_features[a]
        for f in fs:
            # make a new concrete bin from the feature based on mass and RT
            precursor_mass = (f.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos]                        
            concrete_bin = PrecursorBin(k, np.asscalar(precursor_mass), f.rt, f.intensity, within_file_mass_tol, within_file_rt_tol)
            concrete_bin.top_id = tb.bin_id
            concrete_bin.origin = file_idx
            concrete_bin.T = T
            concrete_bin.word_counts = np.zeros(T)
            concrete_bins.append(concrete_bin)
            k += 1

    K = len(concrete_bins)
#     print "File " + str(file_idx) + " has " + str(K) + " concrete bins instantiated"
#     for cb in concrete_bins:
#         print "\t" + str(cb)
    prior_masses = np.array([bb.mass for bb in concrete_bins])[:, None]                # K x 1                                
    prior_rts = np.array([bb.rt for bb in concrete_bins])[:, None]                     # K x 1
    prior_intensities = np.array([bb.intensity for bb in concrete_bins])[:, None]      # K x 1

    # build the matrices for this file
    matRT = sp.lil_matrix((N, K), dtype=np.float)       # N x K, RTs of f n in bin k
    possible = sp.lil_matrix((N, K), dtype=np.int)      # N x K, transformation id+1 of f n in bin k
    transformed = sp.lil_matrix((N, K), dtype=np.float) # N x K, transformed masses of f n in bin k
#     sys.stdout.write("Building matrices for file " + str(file_idx) + " ")
    for n in range(N):
        
        if n%200 == 0:
            sys.stdout.write(str(file_idx))                            
            sys.stdout.flush()

        f = features[n]    
        current_mass, current_rt, current_intensity = f.mass, f.rt, f.intensity
        transformed_masses = (current_mass - adduct_sub)/adduct_mul + adduct_del

        rt_ok = utils.rt_match(current_rt, prior_rts, within_file_rt_tol)
        intensity_ok = (current_intensity <= prior_intensities)
        for t in np.arange(len(transformations)):
            # fill up the target bins that this transformation allows
            mass_ok = utils.mass_match(transformed_masses[t], prior_masses, within_file_mass_tol)
            check = mass_ok*rt_ok*intensity_ok
            pos = np.flatnonzero(check)
            # print (f.feature_id, t, pos)
            possible[n, pos] = t+1
            # and other prior values too
            transformed[n, pos] = transformed_masses[t]
            matRT[n, pos] = current_rt            
            
        possible_clusters = np.nonzero(possible[n, :])[1]
        assert len(possible_clusters) > 0, str(f) + " has no possible clusters"
        
    print
    binning = DiscreteInfo(possible, transformed, matRT, concrete_bins, prior_masses, prior_rts)
    return binning
            
def _make_precursor_bin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
    bin_mass = utils.as_scalar(bin_mass)
    bin_RT = utils.as_scalar(bin_RT)
    bin_intensity = utils.as_scalar(bin_intensity)
    pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
    return pcb
