import os
import sys

import numpy as np


from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.file_binner import _process_file
from discretisation.models import HyperPars
from discretisation.preprocessing import Discretiser

from second_stage_clusterer import DpMixtureGibbs


sys.path.insert(1, os.path.join(sys.path[0], '..'))



def _run_first_stage_clustering(j, peak_data, trans_list, MH, abstract_bins, hp, full_matching):

    binning = _process_file(j, peak_data, abstract_bins, trans_list, MH,
                             hp.within_file_mass_tol, hp.within_file_rt_tol)
    peak_data.set_discrete_info(binning)
            
    print "Clustering file " + str(j) + " by the precursor masses"
    precursorHp = HyperPars()

#     if full_matching:
#         # use the continuous model instead
#         precursorHp.mass_prec = 1.0/(hp.within_file_mass_sd*hp.within_file_mass_sd)
#         precursorHp.rt_prec = 1.0/(hp.within_file_rt_sd*hp.within_file_rt_sd)
#         precursorHp.alpha = hp.alpha_mass
#         precursor_clustering = ContinuousVB(peak_data, precursorHp)
#     else:
#         # use discrete model
#         precursorHp.rt_prec = 1.0/(hp.within_file_rt_sd*hp.within_file_rt_sd)
#         precursorHp.alpha = hp.alpha_mass    
#         precursor_clustering = DiscreteVB(peak_data, precursorHp)                        

    # use the continuous model instead
    precursorHp.mass_prec = 1.0/(hp.within_file_mass_sd*hp.within_file_mass_sd)
    precursorHp.rt_prec = 1.0/(hp.within_file_rt_sd*hp.within_file_rt_sd)
    precursorHp.alpha = hp.alpha_mass
    precursor_clustering = ContinuousVB(peak_data, precursorHp)
    
    precursor_clustering.n_iterations = hp.mass_clustering_n_iterations
    print precursor_clustering
    precursor_clustering.run()
    
    peak_data.remove_discrete_info()    
    return precursor_clustering, binning

def _run_second_stage_clustering(n, top_id, total_topids, data, hp, seed):
    
    selected_rts = data[0]
    selected_word_counts = data[1]
    selected_origins = data[2]
    selected_bins = data[3]
    
    # run dp clustering for each top id
    data = (selected_rts, selected_word_counts, selected_origins)
    dp = DpMixtureGibbs(data, hp, seed=seed)
    dp.nsamps = hp.rt_clustering_nsamps
    dp.burn_in = hp.rt_clustering_burnin
    dp.run() 

    # read the clustering results back
    matching_results = []
    for matched_set in dp.matching_results:
        members = [selected_bins[a] for a in matched_set]
        memberstup = tuple(members)
        matching_results.append(memberstup)

    print "top_id " + str(top_id) + "\t\t(" + str(n) + "/" + str(total_topids) + \
        ")\t\tconcrete_bins=" + str(len(selected_bins)) + "\t\tlast_K = " + str(dp.last_K)
    return matching_results
