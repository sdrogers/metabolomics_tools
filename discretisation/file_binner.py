import numpy as np
import scipy.sparse as sp
import sys

import utils
from models import DiscreteInfo, PrecursorBin

def _process_file(j, peak_data, abstract_bins, transformations, 
                  adduct_sub, adduct_mul, adduct_del, proton_pos, 
                  within_file_mass_tol, within_file_rt_tol):

    concrete_bins = _create_concrete_bins_of_file(j, abstract_bins, transformations, 
                                                  adduct_sub, adduct_mul, adduct_del, proton_pos, 
                                                  within_file_mass_tol, within_file_rt_tol)
    N = len(peak_data.features)     
    K = len(concrete_bins)
    features = peak_data.features    

    prior_masses = np.array([bb.mass for bb in concrete_bins])[:, None]                # K x 1                                
    prior_rts = np.array([bb.rt for bb in concrete_bins])[:, None]                     # K x 1
    prior_intensities = np.array([bb.intensity for bb in concrete_bins])[:, None]      # K x 1

    # build the matrices for this file    
    matRT, possible, transformed = _populate_matrices(j, N, K, 
                                                      prior_masses, prior_rts, prior_intensities,
                                                      features, transformations, 
                                                      adduct_sub, adduct_mul, adduct_del, 
                                                      within_file_mass_tol, within_file_rt_tol)                

    binning = DiscreteInfo(possible, transformed, matRT, concrete_bins, prior_masses, prior_rts)
    return binning

def _create_concrete_bins_of_file(j, abstract_bins, transformations, 
                  adduct_sub, adduct_mul, adduct_del, proton_pos, 
                  within_file_mass_tol, within_file_rt_tol):

    # initialise the 'concrete' realisations of the abstract bins in this file
    T = len(transformations)
    concrete_bins = []
    k = 0
    for bin_id in abstract_bins:
                        
        # get features previously assigned to this abstract bin
        features = abstract_bins[bin_id]
        for f in features:
            # and make a new concrete bin from the feature based on mass and RT
            precursor_mass = (f.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos]                        
            concrete_bin = PrecursorBin(k, np.asscalar(precursor_mass), f.rt, f.intensity, 
                                        within_file_mass_tol, within_file_rt_tol)
            concrete_bin.top_id = bin_id
            concrete_bin.origin = j
            concrete_bin.T = T
            concrete_bin.word_counts = np.zeros(T)
            concrete_bins.append(concrete_bin)
            k += 1
            
    return concrete_bins
            
def _populate_matrices(j, N, K, 
                       prior_masses, prior_rts, prior_intensities,
                       features, transformations, 
                       adduct_sub, adduct_mul, adduct_del, 
                       within_file_mass_tol, within_file_rt_tol):

    matRT = sp.lil_matrix((N, K), dtype=np.float)       # N x K, RTs of f n in bin k
    possible = sp.lil_matrix((N, K), dtype=np.int)      # N x K, transformation id+1 of f n in bin k
    transformed = sp.lil_matrix((N, K), dtype=np.float) # N x K, transformed masses of f n in bin k
    
    for n in range(N):
        
        if n%1000 == 0:
            sys.stdout.write(str(j)+' ')                            
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
    return matRT, possible, transformed
            
def _make_precursor_bin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
    bin_mass = utils.as_scalar(bin_mass)
    bin_RT = utils.as_scalar(bin_RT)
    bin_intensity = utils.as_scalar(bin_intensity)
    pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
    return pcb