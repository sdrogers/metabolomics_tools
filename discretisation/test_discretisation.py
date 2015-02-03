from Annotation import MolAnnotator
from models import FileLoader
import numpy as np
import pylab as plt


def mass_match(m1, m2, tol):
    return np.abs((m1 - m2) / (m1)) < tol * 1e-6

def rt_match(t1, t2, tol):
    return np.abs(t1 - t2) < tol    

def make_mapping(input_file, database_file, transformation_file, mass_tol, rt_tol):

    print 'make_mapping: mass_tol ' + str(mass_tol) + ' rt_tol ' + str(rt_tol)

    # load the std file, database_file molecules and transformation_file
    loader = FileLoader()
    features = loader.load_features(input_file)
    moldb = loader.load_database(database_file)
    transformations = loader.load_transformation(transformation_file)
    
    # The following cell takes all of the M+H peaks and then creates a peak x peak matrix that 
    # hold (for each peak (row)) the precursors that it can be reached. 
    # The values in the matrix are the transformation_file number + 1
    trans_sub = np.array([t.sub for t in transformations])
    trans_mul = np.array([t.mul for t in transformations])
    proton = trans_sub[6]  # 6 is the index of M+H in mulsub.txt
    masses = np.array([f.mass for f in features])
    precursor_masses = masses - proton 
    rts = np.array([f.rt for f in features])

    mapping = np.zeros((len(features), len(features)))
    for i in np.arange(len(features)):
        peak_rt = rts[i]
        # True if this peak's RT is close enough to the RTs of the initial peaks that generated the M+H bins 
        rt_ok = rt_match(peak_rt, rts, rt_tol)
        for j in np.arange(len(transformations)):
            trans_mass = (masses[i] - trans_sub[j]) / trans_mul[j]
            # True if this peak's mass before transformation_file is close enough to the precursor masses from M+H
            mass_ok = mass_match(precursor_masses, trans_mass, mass_tol)
            # combine the results
            matching = np.logical_and(mass_ok, rt_ok)         
            # find the indices of the True entries only
            temp = matching.nonzero() 
            q = temp[0]
            mapping[i][q] = j + 1

    # compare annotations of database_file molecules in the continous and discrete cases
    ann = MolAnnotator()
    ann.annotate_mols(moldb, precursor_masses, mass_tol)
    
    return mapping
    
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
    plt.title('mapping')
    plt.suptitle(filename)
    plt.show()    
    
def main():

    # basedir = '/home/joewandy/git/metabolomics_tools/discretisation'
    basedir = '.'
    database = basedir + '/database/std1_mols.csv'
    transformation = basedir + '/mulsub.txt'
    filename = basedir + '/input/std1_csv/std1-file1.identified.csv'

    mass_tol = 2
    rt_tol = 999
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)
    print

    mass_tol = 2
    rt_tol = 30
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)
    print

    mass_tol = 2
    rt_tol = 10
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)
    
    plt.show()

if __name__ == "__main__": main()
