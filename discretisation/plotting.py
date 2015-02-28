from collections import Counter
from matplotlib import pylab as plt
import numpy as np

def print_cluster_size(cluster_size, samp):
    print "Sample " + str(samp) + " biggest cluster: " + str(cluster_size.max()) + " (" + str(cluster_size.argmax()) + ")"

def print_cluster_sizes(bins, samp, time_taken, is_sample):
    cluster_sizes = [str(mb.get_features_count()) for mb in bins]
    c = Counter()
    c.update(cluster_sizes) # count frequencies of each size 
    if is_sample:
        print('SAMPLE %d %4.2fs\t%s' % ((samp+1), time_taken, str(c)))
    else:
        print('BURN %d %4.2fs\t%s' % ((samp+1), time_taken, str(c)))
        
def print_last_sample(bins, feature_annotation):
    print 'Last sample report'
    count_empty_bins = 0
    bin_mols = []
    bin_mols_unique = set()
    
    # sort the bin by no. of features first (biggest first)
    bins.sort(key=lambda x: x.get_features_count(), reverse=True)
    # loop through bins and print stuff
    for mass_bin in bins:
        # count molecules annotated to bin
        bin_mols.extend(mass_bin.molecules)
        bin_mols_unique.update(mass_bin.molecules)
        if mass_bin.get_features_count() > 0:
            # print header
            print mass_bin
            for mol in mass_bin.molecules:
                print "\t" + str(mol)
            table = []
            table.append(['feature_id', 'mass', 'rt', 'intensity', 'annotation', 'gt_metabolite', 'gt_adduct'])
            # print features in this mass bin
            for f in mass_bin.features:
                table.append([str(f.feature_id), str(f.mass), str(f.rt), str(f.intensity), 
                              feature_annotation[f], str(f.gt_metabolite), str(f.gt_adduct)])
            _print_table(table)
            print
        else:
            count_empty_bins = count_empty_bins + 1

    print 'Empty bins=' + str(count_empty_bins)
    print 'Occupied bins=' + str(len(bins) - count_empty_bins) 

def _print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print "| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |"
                                
class ClusterPlotter(object):
    # an uncommented class for plotting clusters
    def __init__(self,peak_data,cluster_model):
        self.cluster_model = cluster_model
        self.peak_data = peak_data
        self.cluster_membership = (cluster_model.Z>0.5)

    def summary(self):
        print "Cluster output"
        s = self.cluster_membership.sum(0)
        nnz = (s>0).sum()
        print "Number of non-empty clusters: " + str(nnz) + " (of " + str(s.size) + ")"
        si = (self.cluster_membership).sum(0)
        print
        print "Size: count"
        for i in np.arange(0,si.max()+1):
            print str(i) + ": " + str((si==i).sum())
        t = (self.peak_data.possible.multiply(self.cluster_membership)).data
        t -= 1
        print
        print "Trans: count"
        for i in np.arange(len(self.peak_data.transformations)):
            print self.peak_data.transformations[i].name + ": " + str((t==i).sum())


    def plot_biggest(self,n_plot):
        # plots the n_plot biggest clusters
        s = self.cluster_membership.sum(0)
        order = np.argsort(s)
        
        for i in np.arange(s.size-1,s.size-n_plot-1,-1):
            cluster = order[0,i]
            peaks = np.nonzero(self.cluster_membership.getcol(cluster))[0]
            plt.figure(figsize=(8,8))
            plt.subplot(1,2,1)
            plt.plot(self.peak_data.mass[peaks],self.peak_data.rt[peaks],'ro')
            plt.plot(self.peak_data.transformed[peaks,cluster].toarray(),self.peak_data.rt[peaks],'ko')

            plt.subplot(1,2,2)
            for peak in peaks:
                plt.plot((self.peak_data.mass[peak], self.peak_data.mass[peak]),(0,self.peak_data.intensity[peak]))
                tr = self.peak_data.possible[peak,cluster]-1
                plt.text(self.peak_data.mass[peak],self.peak_data.intensity[peak],self.peak_data.transformations[tr].name)
                title_string = "Mean RT: " + str(self.cluster_model.cluster_rt_mean[cluster]) + "(" + \
                    str(1.0/self.cluster_model.cluster_rt_prec[cluster]) + ")"                    
                if hasattr(self.cluster_model, 'cluster_mass_mean'):
                    title_string += " Mean Mass: " + str(self.cluster_model.cluster_mass_mean[cluster]) + \
                        "(" + str(1.0/self.cluster_model.cluster_mass_prec[cluster]) + ")"
                plt.title(title_string) 
        plt.show()

        
    def intensity_plot(self):
        # This will create the plot of intensity ratios versus intensity ratios
        print "hello"