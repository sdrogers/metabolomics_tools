from identification import MolAnnotator
from preprocessing import FileLoader
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

    basedir = '.'

    database = basedir + '/database/std1_mols.csv'
    transformation = basedir + '/mulsubs/mulsub.txt'
    input_file = basedir + '/input/std1_csv_subset'
    mass_tol = 2
    rt_tol = 5

    # load the std file, database_file molecules and transformation_file
    loader = FileLoader()
    data_list = loader.load_model_input(input_file, database, transformation, mass_tol, rt_tol)
    peak_data = data_list[0]
    plot_hist(peak_data.possible, input_file, mass_tol, rt_tol)

    # try identify
#     ann = MolAnnotator()    
#     moldb = peak_data.database
#     prior_masses = peak_data.prior_masses    
#     bins = peak_data.bins
#     ann.identify_normal(moldb, prior_masses, mass_tol)
#     ann.identify_bins(moldb, bins)

if __name__ == "__main__": main()