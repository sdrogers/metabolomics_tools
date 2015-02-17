from identification import MolAnnotator
from models import FileLoader
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
    plt.imshow(mapping)
    plt.title('possible')
    plt.suptitle(filename)
    plt.show()    
    
def main():

    basedir = '.'

    database = basedir + '/database/std1_mols.csv'
    transformation = basedir + '/mulsubs/mulsub.txt'
    input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'
    mass_tol = 3
    rt_tol = 30

    # load the std file, database_file molecules and transformation_file
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database, transformation, mass_tol, rt_tol)
    plot_hist(peak_data.possible.todense(), input_file, mass_tol, rt_tol)

    # try identify
    ann = MolAnnotator()
    
    moldb = peak_data.database
    precursor_masses = peak_data.precursor_mass    
    bins = peak_data.bins

    ann.identify_normal(moldb, precursor_masses)

    ann.identify_bins(moldb, bins)

if __name__ == "__main__": main()