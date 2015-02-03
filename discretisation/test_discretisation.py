import numpy as np
import pylab as plt
import csv
from models import *
from Annotation import MolAnnotator

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def mass_match(m1, m2, tol):
    return np.abs((m1-m2)/(m1))<tol*1e-6

def rt_match(t1, t2, tol):
    return np.abs(t1-t2)<tol    

def make_mapping(input_file, database, transformation, mass_tol, rt_tol):

    print 'make_mapping: mass_tol ' + str(mass_tol) + ' rt_tol ' + str(rt_tol)

    # load the std file
    peakid = []
    mass = []
    rt = []
    intensity = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=':')
        next(reader, None)  # skip the headers
        for elements in reader:
            peakid.append(num(elements[0]))
            mass.append(num(elements[1]))
            rt.append(num(elements[2]))
            intensity.append(num(elements[3])) 
    
    # load the actual molecules
    moldb = []
    with open(database, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for elements in reader:
            mol = DatabaseEntry(id=elements[0], name=elements[1], formula=elements[2], mass=num(elements[3]))
            moldb.append(mol)
    
    # load transformations
    trans_names = []
    trans_sub = []
    trans_mul = []
    with open(transformation, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for elements in reader:
            trans_names.append(elements[0])
            trans_sub.append(num(elements[1]))
            trans_mul.append(num(elements[2]))
    trans_sub = np.array(trans_sub)
    trans_mul = np.array(trans_mul)
    
    # The following cell takes all of the M+H peaks and then creates a peak x peak matrix that 
    # hold (for each peak (row)) the precursors that it can be reached. 
    # The values in the matrix are the transformation number + 1
    proton = trans_sub[6]
    precursor_masses = np.array(mass) - proton # precursor masses from each peak after substract M+H
    all_rts = np.array(rt) # initial RTs of all peaks
    n_peaks = precursor_masses.size
    mapping = np.zeros((n_peaks, n_peaks))
    n_trans = len(trans_names)
    for i in np.arange(n_peaks):
        peak_rt = rt[i]
        # True if this peak's RT is close enough to the RTs of the initial peaks that generated the M+H bins 
        rt_ok = rt_match(peak_rt, all_rts, rt_tol)
        for j in np.arange(n_trans):
            trans_mass = (mass[i] - trans_sub[j])/trans_mul[j]
            # True if this peak's mass before transformation is close enough to the precursor masses from M+H
            mass_ok = mass_match(precursor_masses, trans_mass, mass_tol)
            # combine the results
            matching = np.logical_and(mass_ok, rt_ok)         
            # find the indices of the True entries only
            temp = matching.nonzero() 
            q = temp[0]
            mapping[i][q] = j+1

    # compare annotations of database molecules in the continous and discrete cases
    ann = MolAnnotator()
    ann.annotate_mols(moldb, precursor_masses, mass_tol)
    
    return mapping
    
# We can histogram the number of transformations available for each peak. mini_hist holds this. 
# Note that all peaks have >0 transformations as each peak's precursor is in the list
def plot_hist(mapping, filename, mass_tol, rt_tol):
    no_trans = (mapping>0).sum(1)
    mini_hist = []
    for i in np.arange(10)+1:
        mini_hist.append((no_trans==i).sum())
    print 'mini_hist ' + str(mini_hist)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(10)+1, mini_hist)
    title = 'MASS_TOL ' + str(mass_tol) + ', RT_TOL ' + str(rt_tol)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.imshow(mapping)
    plt.title('mapping')
    plt.suptitle(filename)
    plt.show()    
    
def main():

    basedir = '/home/joewandy/git/metabolomics_tools/discretisation'
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
