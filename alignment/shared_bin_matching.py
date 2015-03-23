import itertools

from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.models import HyperPars
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
from dp_rt_clusterer import DpMixtureGibbs
import numpy as np
import pylab as plt
import scipy.sparse as sp

def plot_hist(mapping, filename, mass_tol, rt_tol):
    no_trans = (mapping > 0).sum(1)
    mini_hist = []
    for i in np.arange(10) + 1:
        mini_hist.append((no_trans == i).sum())
    print 'mini_hist ' + str(mini_hist)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(10) + 1, mini_hist)
    title = 'MASS_TOL ' + str(mass_tol) + ', RT_TOL ' + str(rt_tol)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.spy(mapping, markersize=1)
    plt.title('possible')
    plt.suptitle(filename)
    plt.show()        
    
def main():

    basedir = '../discretisation'
    database = basedir + '/database/std1_mols.csv'
    transformation = basedir + '/mulsubs/mulsub2.txt'
    input_file = basedir + '/input/std1_csv_subset'
    
    binning_mass_tol = 2.0                  # mass tolerance in ppm when binning
    binning_rt_tol = 5.0                    # rt tolerance in seconds when binning
    within_file_rt_sd = binning_rt_tol/2    # standard deviation of each cluster when clustering by precursor masses in a single file
    across_file_rt_sd = 5.0                 # standard deviation of mixture component when clustering by RT across files
    alpha_mass = 10000.0                    # alpha for precursor mass clustering
    alpha_rt = 10000.0                      # alpha for DP mixture
    t = 0.30                                # threshold for everything
    limit_n = -1                            # the number of features to load per file, -1 to load all

    # First stage clustering. 
    # Here we cluster peak features by their precursor masses to the common bins shared across files.
    loader = FileLoader()
    data_list = loader.load_model_input(input_file, database, transformation, binning_mass_tol, binning_rt_tol, limit_n=limit_n)
    all_bins = []
    posterior_bin_rts = []
    
    for j in range(len(data_list)):

        # run precursor mass clustering
        peak_data = data_list[j]
        plot_hist(peak_data.possible, input_file, binning_mass_tol, binning_rt_tol)
        print "Clustering file " + str(j) + " by precursor masses"
        hp = HyperPars()
        hp.rt_prec = 1.0/(within_file_rt_sd*within_file_rt_sd)
        hp.alpha = alpha_mass
        discrete = DiscreteVB(peak_data, hp)
        discrete.n_iterations = 20
        print discrete
        discrete.run()

        # pick the non-empty bins for the second stage clustering
        cluster_membership = (discrete.Z>t)
        s = cluster_membership.sum(0)
        nnz_idx = s.nonzero()[1]  
        nnz_idx = np.squeeze(np.asarray(nnz_idx)) # flatten the thing
        
        # find the non-empty bins
        bins = [peak_data.bins[a] for a in nnz_idx]
        all_bins.extend(bins)
        
        # find the non-empty bins' posterior RT values
        bin_rts = discrete.cluster_rt_mean[nnz_idx]        
        bin_rts = bin_rts.ravel().tolist()
        posterior_bin_rts.extend(bin_rts)
        
        # assign peaks into their respective bins
        # note: a peak can belong to multiple bins, depending on the choice of threshold t
        cx = cluster_membership.tocoo()
        for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
            f = peak_data.features[i]
            bb = peak_data.bins[j] # copy of the common bin specific to file j
            bb.add_feature(f)
        
        # make some plots
        cp = ClusterPlotter(peak_data, discrete)
        cp.summary()
        # cp.plot_biggest(3)
    
    # Second-stage clustering
    N = len(all_bins)
    assert N == len(posterior_bin_rts)
    
    # Here we cluster the 'concrete' common bins across files by their posterior RT values
    hp = HyperPars()
    hp.rt_prec = 1.0/(across_file_rt_sd*across_file_rt_sd)
    hp.rt_prior_prec = 5E-3
    hp.alpha = alpha_rt
    data = (posterior_bin_rts, all_bins)
    dp = DpMixtureGibbs(data, hp)
    dp.nsamps = 20
    dp.burn_in = 0
    dp.run()
    
    # only bins with the same id can be matched across files
#     mask = sp.lil_matrix((N, N), dtype=np.float)
#     for n1 in range(N):
#         for n2 in range(N):
#             bin1 = all_bins[n1]
#             bin2 = all_bins[n2]
#             if bin1.bin_id == bin2.bin_id and bin1.origin != bin2.origin:
#                 mask[n1, n2] = 1
#     plt.imshow(mask.todense(), cmap='cool')
#     plt.show()    
#     ZZ_all = dp.ZZ_all.multiply(mask)

    plt.spy(ZZ_all, markersize=1)
    plt.show()
    x = []
    cx = ZZ_all.tocoo()    
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        x.append(v)       
    x = np.array(x) 
    plt.hist(x, 10)
    plt.title('Distribution of values in ZZ_all')
    plt.xlabel('Probabilities')
    plt.ylabel('Count')
    plt.show()        
    
if __name__ == "__main__": main()