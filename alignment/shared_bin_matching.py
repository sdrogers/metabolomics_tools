from discretisation.discrete_mass_clusterer import DiscreteVB
from discretisation.identification import MolAnnotator
from discretisation.models import HyperPars
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader
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

    # first stage clustering of peak features to some common bins that are shared across files
    loader = FileLoader()
    data_list = loader.load_model_input(input_file, database, transformation, mass_tol, rt_tol)
    for j in range(len(data_list)):
        peak_data = data_list[j]
        plot_hist(peak_data.possible, input_file, mass_tol, rt_tol)
        print "Clustering file " + str(j) + " by precursor masses"
        hp = HyperPars()
        n_iters = 20    
        discrete = DiscreteVB(peak_data, hp)
        discrete.n_iterations = n_iters
        print discrete
        discrete.run()        
        cp = ClusterPlotter(peak_data, discrete)
        cp.summary()
        cp.plot_biggest(3)
    
    # second-stage clustering
    # first we need to define the 'abstract' bins
    
    # then cluster 'concrete' bins by RT values and their possible assignments to the abstract bins 

if __name__ == "__main__": main()