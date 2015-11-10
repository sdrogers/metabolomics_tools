import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from discretisation.adduct_cluster import AdductCluster, Peak
from second_stage_clusterer import DpMixtureGibbs

def _run_first_stage_clustering(j, peak_data, hp, trans_filename):

    sys.stdout.flush()
    ac = AdductCluster(mass_tol=hp.within_file_mass_tol, rt_tol=hp.within_file_rt_tol, 
                       alpha=hp.alpha_mass, mh_biggest=True, transformation_file=trans_filename, verbose=2)

    peak_list = peak_data.features
    ac.init_from_list(peak_list)

    ac.init_vb()
    for n in range(hp.mass_clustering_n_iterations):
        print "VB step %d file %d " % (n, j)
        sys.stdout.flush()
        ac.vb_step()

    return ac

def _run_second_stage_clustering(n, cluster_list, hp, seed):
    
    if seed == -1:
        seed = 1234567890
    
    rts = []
    word_counts = []
    origins = []
    for cluster in cluster_list:
        rts.append(cluster.mu_rt)
        word_counts.append(cluster.word_counts)
        origins.append(cluster.origin)
    data = (rts, word_counts, origins)
    
    # run dp clustering for each top id
    dp = DpMixtureGibbs(data, hp, seed=seed)
    dp.nsamps = hp.rt_clustering_nsamps
    dp.burn_in = hp.rt_clustering_burnin
    dp.run() 

    # read the clustering results back
    matching_results = []
    for matched_set in dp.matching_results:
        members = [cluster_list[a] for a in matched_set]
        memberstup = tuple(members)
        matching_results.append(memberstup)

    print "n " + str(n) + "\tcluster_list=" + str(len(cluster_list)) + "\tlast_K = " + str(dp.last_K)
    return matching_results