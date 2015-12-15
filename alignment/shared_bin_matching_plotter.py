import itertools

import numpy as np
import pylab as plt

def plot_bin_posterior_rt(bin_rts, j):
    plt.figure()
    plt.plot(bin_rts, '.b')
    plt.title("Posterior RT values for file " + str(j))
    plt.xlabel("Non-empty bins")
    plt.ylabel("RT")
    plt.show()    

def plot_possible_hist(mapping, filename, mass_tol, rt_tol):
    no_trans = (mapping > 0).sum(1)
    mini_hist = []
    for i in np.arange(10) + 1:
        mini_hist.append((no_trans == i).sum())
    print 'mini_hist ' + str(mini_hist)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(10) + 1, mini_hist)
    title = 'Binning -- MASS_TOL ' + str(mass_tol) + ', RT_TOL ' + str(rt_tol)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.spy(mapping, markersize=1)
    plt.title('possible')
    plt.suptitle(filename)
    plt.show()     

def plot_bin_vs_bin(file_bins, file_post_rts):

    first_bins = file_bins[0]
    first_rts = file_post_rts[0]
    second_bins = file_bins[1]
    second_rts = file_post_rts[1]
    xs = []
    ys = []
    for j1 in range(len(first_bins)):
        bin1 = first_bins[j1]
        j2s, bin2s = find_same_top_id(bin1, second_bins)
        for j2 in j2s:        
            rt1 = first_rts[j1]
            rt2 = second_rts[j2]
            xs.append(rt1)
            ys.append(rt2)

    plt.figure()
    plt.plot(np.array(xs), np.array(ys), '.b')
    plt.xlabel("File 0")
    plt.ylabel("File 1")
    plt.title("Bin-vs-bin posterior RTs")
    plt.show()
     
    sizes = []
    for bin1 in first_bins:
        sizes.append(bin1.get_features_count())
    plt.figure()
    plt.plot(np.array(sizes), 'r.')
    sizes = []
    for bin2 in second_bins:
        sizes.append(bin2.get_features_count())
    plt.plot(np.array(sizes), 'g.')
    plt.title("Bin sizes in file 0 & 1")
    plt.xlabel("Bins")
    plt.ylabel("Sizes")
    plt.show()
    
def find_same_top_id(self, key, items):
    indices = []
    results = []
    for a in range(len(items)):
        check = items[a]
        if key.top_id == check.top_id:
            indices.append(a)
            results.append(check)
    return indices, results
        
def plot_ZZ_all(ZZ_all):
    
    # plot distribution of values in ZZ_all
    ZZ_all = ZZ_all
    x = []
    cx = ZZ_all.tocoo()    
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        x.append(v)       
    x = np.array(x)
    plt.figure() 
    plt.hist(x, 10)
    plt.title("DP RT clustering -- ZZ_all")
    plt.xlabel("Probabilities")
    plt.ylabel("Count")
    plt.show()        

def plot_aligned_peaksets_probabilities(probs):
    plt.figure()
    plt.hist(probs, 20)
    plt.title("Aligned peaksets probabilities")
    plt.xlabel("Probabilities")
    plt.ylabel("Count")
    plt.show()