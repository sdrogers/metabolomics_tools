import numpy as np
import pylab as plt
import csv

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

    print 'make_mapping: ' + input_file + database + transformation + ' mass_tol ' + str(mass_tol) + ' rt_tol ' + str(rt_tol)

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
    molid = []
    molname = []
    molformula = []
    molmass = []
    with open(database, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for elements in reader:
            molid.append(elements[0])
            molname.append(elements[1])
            molformula.append(elements[2])
            molmass.append(num(elements[3]))
    
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
    
    # The following cell takes all of the M+H peaks and then creates a peak x peak matrix that hold (for each peak (row)) the precursors that it can be reached. 
    # The values in the matrix are the transformation number + 1
    proton = trans_sub[6]
    precursor_masses = np.array(mass) - proton # precursor masses from each peak after substract M+H
    all_rts = np.array(rt) # initial RTs of all peaks
    n_peaks = precursor_masses.size
    mapping = np.zeros((n_peaks, n_peaks))
    n_trans = len(trans_names)
    for i in np.arange(n_peaks):
        for j in np.arange(n_trans):
            trans_mass = (mass[i] - trans_sub[j])/trans_mul[j]
            peak_rt = rt[i]
            # True if this peak's mass after transformation is close enough to the precursor masses
            mass_ok = mass_match(trans_mass, precursor_masses, mass_tol)
            # True if this peak's RT is close enough to the RTs of other peaks
            rt_ok = rt_match(peak_rt, all_rts, rt_tol)
            # combine the results
            matching = np.logical_and(mass_ok, rt_ok)         
            # find the indices of the True entries only
            temp = matching.nonzero() 
            q = temp[0]
            mapping[i][q] = j+1
            
    return mapping

# We can histogram the number of transformations available for each peak. mini_hist holds this. 
# Note that all peaks have >0 transformations as each peak's precursor is in the list
def plot_hist(mapping, filename, mass_tol, rt_tol):
    title = filename + '\nMASS_TOL ' + str(mass_tol) + ', RT_TOL ' + str(rt_tol)
    no_trans = (mapping>0).sum(1)
    mini_hist = []
    for i in np.arange(10)+1:
        mini_hist.append((no_trans==i).sum())
    print mini_hist
    plt.figure()
    plt.bar(np.arange(10)+1, mini_hist)
    plt.title(title)
    plt.show(block=False)
    
def main():

    database = 'database/std1_mols.csv'
    transformation = 'mulsub.txt'

    filename = 'input/std1_csv/std1-file1.identified.csv'
    mass_tol = 2
    rt_tol = 999
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)

    filename = 'input/std1_csv/std1-file1.identified.csv'
    mass_tol = 2
    rt_tol = 30
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)

    filename = 'input/std1_csv/std1-file1.identified.csv'
    mass_tol = 2
    rt_tol = 10
    mapping = make_mapping(filename, database, transformation, mass_tol, rt_tol)
    plot_hist(mapping, filename, mass_tol, rt_tol)
    
    plt.show()

if __name__ == "__main__": main()