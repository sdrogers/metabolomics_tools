""" 
We pose the following question:

*Q1. Is aligning peaks at feature-level better/worse than aligning at group-level?*

We can try the following experiment to answer the question:
<pre>
    (a) Take any two files. For each peak a in file 1, find out how many peaks in file 2 can potentially 
        be matched within some mass & RT windows from peak a (count the # of candidate matches).
        Gradually increase the RT window and see what happens.
    (b) Repeat for group-level, i.e. do clustering based on the mass or other clustering methods, 
        set threshold t on the output of peak-vs-peak matrix to be in the same cluster, 
        consider everything >t to be in the same cluster.
</pre>

In short, **approach (a)** is to do alignment on feature-level, while **approach (b)** is operating on the group-level.
"""

import numpy as np
from discretisation import models
import pylab as plt

import sys
sys.path.append('..')
from discretisation import models

def make_boxplot(filenames, mass_tol, rt_tols, title):
    # load the data
    loader = models.FileLoader()
    all_data = []
    for j in range(len(filenames)):
        fn = filenames[j]
        peak_data = loader.load_model_input(fn, '', '', 0, 0, discretise=False)
        print 'Read ' + str(len(peak_data.features)) + ' features from ' + fn
        all_data.append(peak_data)
        
    # for the different rt tolerances ...
    features_1 = all_data[0].features
    features_2 = all_data[1].features
    ax1 = None
    for t in range(len(rt_tols)):
        rt_tol = rt_tols[t]
        print 'Processing rt_tol=' + str(rt_tol)
        # enumerate the no. of potential matches for each feature
        num_matches = []
        for f1 in features_1:
            mass_ok = all_data[1].mass_match(f1.mass, all_data[1].mass, mass_tol)
            rt_ok = all_data[1].rt_match(f1.rt, all_data[1].rt, rt_tol)
            pos = np.flatnonzero(rt_ok*mass_ok)
            count = len(pos)
            num_matches.append(count)
        # share the axis for subplots
        if ax1 is None:
            ax1 = plt.subplot(1, len(rt_tols), t+1)
            plt.ylabel('# potential matches')
        else:
            plt.subplot(1, len(rt_tols), t+1, sharey=ax1)
        # make boxplot and show median, mean of potential matches
        plt.boxplot(num_matches)
        plt.xticks([1], ['Features'])
        plt.title('rt_tol=' + str(rt_tol))
        mean = np.mean(np.array(num_matches))
        median = np.median(np.array(num_matches))
        std = np.std(np.array(num_matches))        
        print("\tmean={:.2f}".format(mean))
        print("\tmedian={:.2f}".format(median))
        print("\tstd={:.2f}".format(std))
        plt.suptitle('# matches per feature (' + title + '), mass_tol=' + str(mass_tol))
    plt.show()
    
mass_tol = 10
rt_tols = [10, 30, 60, 120]

# load std1 pos
basedir = 'input/std1_csv'
filenames = [ 
    basedir + '/std1-file1.identified.csv', 
    basedir + '/std1-file2.identified.csv' 
]
label = "STD1 POS"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print '---------------------------------------------------------------------'
print

# load std3 pos
basedir = 'input/std3_csv'
filenames = [ 
    basedir + '/std3-file1.identified.csv', 
    basedir + '/std3-file2.identified.csv' 
]
label = "STD3 POS"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print

''' mass_tol is set based on inspection of the ground truth for M1 & M2, 
e.g. 
111.0096, 111.0126, 111.0118
873.5103, 873.5104, 873.5085
374.1745, 374.1702, 374.1752 
'''
mass_tol = 1000
rt_tols = [10, 30, 60, 120]

# load M1
basedir = 'input/M1'
filenames = [ 
    basedir + '/M1_1.txt', 
    basedir + '/M1_2.txt' 
]
label = "M1"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print '---------------------------------------------------------------------'
print

# load M2
basedir = 'input/M2'
filenames = [ 
    basedir + '/M2_1.txt', 
    basedir + '/M2_2.txt' 
]
label = "M2"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print

''' 
based on inspection of the ground truth, 
e.g. 
698.61, 698.466
943.654, 943.334
870.956, 870.418
643.473, 643.348
843.657, 844.532, 844.813
'''
mass_tol = 10000
rt_tols = [60, 120, 180, 240]

# load P1
basedir = 'input/P1/000'
filenames = [ 
    basedir + '/021010_jp32A_15ul_1_000_ld_020.txt', 
    basedir + '/021016_jp32A_10ul_3_000_ld_020.txt' 
]
label = "P1"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print '---------------------------------------------------------------------'
print

# load P2
basedir = 'input/P2/000'
filenames = [ 
    basedir + '/6-06-03_000.txt', 
    basedir + '/6-17-03_000.txt' 
]
label = "P2"
print label + ", mass_tol=" + str(mass_tol)
make_boxplot(filenames, mass_tol, rt_tols, label)
print