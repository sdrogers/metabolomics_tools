from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.models import HyperPars
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
from dp_rt_clusterer import DpMixtureGibbs
import numpy as np
import pylab as plt


# We can histogram the number of transformations available for each peak. mini_hist holds this. 
# Note that all peaks have >0 transformations as each peak's precursor is in the list
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
    transformation = basedir + '/mulsubs/mulsub.txt'
    input_file = basedir + '/input/std1_csv_subset'
    mass_tol = 2
    rt_tol = 5
    across_file_sd = 5
    dp_alpha = 100
    t = 0.90
    limit_n = 1000

    # first stage clustering of peak features to some common bins that are shared across files
    loader = FileLoader()
    data_list = loader.load_model_input(input_file, database, transformation, mass_tol, rt_tol, limit_n=limit_n)
    bin_rts = []
    for j in range(len(data_list)):
        peak_data = data_list[j]
        plot_hist(peak_data.possible, input_file, mass_tol, rt_tol)
        print "Clustering file " + str(j) + " by precursor masses"
        hp = HyperPars()
        discrete = DiscreteVB(peak_data, hp)
        discrete.n_iterations = 20
        print discrete
        discrete.run()
        # pick the non-empty bins for the second stage clustering
        cluster_membership = (discrete.Z>t)
        s = cluster_membership.sum(0)
        nnz_idx = s.nonzero()[1]  
        cluster_rt_mean = discrete.cluster_rt_mean[nnz_idx]        
        flattened = cluster_rt_mean.ravel().tolist()
        bin_rts.extend(flattened)
        cp = ClusterPlotter(peak_data, discrete)
        cp.summary()
#         cp.plot_biggest(3)
    
    # second-stage clustering
    data = np.array(bin_rts)
#     print "Cluster RT means"
#     plt.figure()
#     plt.plot(data, 'b.', markersize=1)
#     plt.show()
    hp = HyperPars()
    hp.rt_prec = 1.0/(across_file_sd*across_file_sd)
    hp.rt_prior_prec = 5E-3
    hp.alpha = dp_alpha
    dp = DpMixtureGibbs(data, hp)
    dp.nsamps = 100
    dp.burn_in = 0
    dp.run()
    if limit_n > 1000:
        plt.spy(dp.ZZ_all>t, markersize=1)
    else:
        plt.imshow(dp.ZZ_all.todense(), cmap='cool')
        plt.colorbar()
    plt.show()
    
if __name__ == "__main__": main()