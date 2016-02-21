
import operator
import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from adduct_cluster import AdductCluster, Peak
from second_stage_clusterer import DpMixtureGibbs

def _run_first_stage_clustering(j, peak_data, hp, trans_filename, mh_biggest, use_vb):

    sys.stdout.flush()
    ac = AdductCluster(mass_tol=hp.within_file_mass_tol, rt_tol=hp.within_file_rt_tol, 
                       alpha=hp.alpha_mass, mh_biggest=mh_biggest, transformation_file=trans_filename, verbose=2)

    peak_list = peak_data.features
    ac.init_from_list(peak_list)

    if use_vb:
        ac.init_vb()
        for n in range(hp.mass_clustering_n_iterations):
            print "VB step %d file %d " % (n, j)
            sys.stdout.flush()
            ac.vb_step()
    else:
        ac.multi_sample(hp.mass_clustering_n_iterations)
        ac.compute_posterior_probs()    

    return ac

def _run_second_stage_clustering(n, cluster_list, hp, seed, verbose=False):
    
    if seed == -1:
        seed = 1234567890
    
    masses = []
    rts = []
    word_counts = []
    origins = []
    for cluster in cluster_list:
        masses.append(cluster.mu_mass)
        rts.append(cluster.mu_rt)
        word_counts.append(cluster.word_counts)
        origins.append(cluster.origin)
    data = (masses, rts, word_counts, origins)
    
    # run dp clustering for each top id
    dp = DpMixtureGibbs(data, hp, seed=seed, verbose=verbose)
    dp.nsamps = hp.rt_clustering_nsamps
    dp.burn_in = hp.rt_clustering_burnin
    dp.run() 

    # read the clustering results back
    matching_results = []
    results = {}
    for matched_set in dp.matching_results:
        members = [cluster_list[a] for a in matched_set]
        memberstup = tuple(members)
        matching_results.append(memberstup)
        if matched_set in results:
            results[matched_set] += 1
        else:
            results[matched_set] = 1

    output = "n " + str(n) + "\tcluster_list=" + str(len(cluster_list)) + "\tlast_K = " + str(dp.last_K)
    if verbose:
        output += "\n"
        mass_list = dp.masses.tolist()
        rt_list = dp.rts.tolist()
        adduct_list = dp.word_counts_list
        for i in range(len(mass_list)):
            output += "#" + str(i) + "\n"
            output += "  mass = %.5f rt=%.3f\n" % (mass_list[i], rt_list[i])
            output += "  adducts = %s\n" % (adduct_list[i].tolist())
        output += "clustering results\n"
        sorted_results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)    
        for key, value in sorted_results:
            output += "  %s = %d\n" % (key, value)
    print output
    sys.stdout.flush()
    
    return matching_results