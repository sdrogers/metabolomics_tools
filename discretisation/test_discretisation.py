import numpy as np
import pylab as plt

STD = 'std1' # which standard set
FILEIDX = 1 # which file in the standard set

FILENAME = 'input/' + STD + '_csv/' + STD + '-file' + str(FILEIDX) + '.identified.csv'
DATABASE = 'database/' + STD + '_mols.csv'

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def match(m1,m2,tol):
    return np.abs((m1-m2)/(m1))<tol*1e-6

# load the std file
f = open(FILENAME)
heads =f.readline()
pos = 0
peakid = []
mass = []
rt = []
intensity = []
for line in f:
    elements = line.split(':')
    peakid.append(num(elements[0]))
    mass.append(num(elements[1]))
    rt.append(num(elements[2]))
    intensity.append(num(elements[3])) 
f.close()

# load the actual molecules
molsFile = open(DATABASE)
molid = []
molname = []
molformula = []
molmass = []
for line in molsFile:
    elements = line.split(',')
    molid.append(elements[0])
    molname.append(elements[1])
    molformula.append(elements[2])
    molmass.append(num(elements[3]))
molsFile.close()

# load transformations
trans_names = []
trans_sub = []
trans_mul = []
trans_file = open('mulsub.txt')
for line in trans_file:
    elements = line.split(',')
    trans_names.append(elements[0])
    trans_sub.append(num(elements[1]))
    trans_mul.append(num(elements[2]))
trans_file.close()
trans_sub = np.array(trans_sub)
trans_mul = np.array(trans_mul)

# The following cell takes all of the M+H peaks and then creates a peak x peak matrix that hold (for each peak (row)) the precursors that it can be reached. The values in the matrix are the transformation number + 1
proton = trans_sub[6]
precursor_masses = np.array(mass) - proton
n_peaks = precursor_masses.size
mapping = np.zeros((n_peaks,n_peaks))
n_trans = len(trans_names)
for i in np.arange(n_peaks):
    for j in np.arange(n_trans):
        trans_mass = (mass[i] - trans_sub[j])/trans_mul[j]
        mass_matching = match(trans_mass,precursor_masses,2)
        temp = mass_matching.nonzero() # find all the True entries
        q = temp[0]
        mapping[i][q] = j+1

no_trans = (mapping>0).sum(1)

# We can histogram the number of transformations available for each peak. mini_hist holds this. Note that all peaks have >0 transformations as each peak's precursor is in the list
mini_hist = []
for i in np.arange(10)+1:
    mini_hist.append((no_trans==i).sum())
print mini_hist
plt.bar(np.arange(10)+1,mini_hist)
plt.show()

