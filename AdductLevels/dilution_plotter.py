from collections import Counter

from matplotlib.font_manager import FontProperties

from discretisation.preprocessing import FileLoader
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import pylab as plt


class DilutionPlotter(object):
    
    def __init__(self, input_file):

        self.stdfile = input_file
        loader = FileLoader()
        self.transformations = loader.load_transformation('mulsubs/mulsub2.txt')
        self.database = loader.load_database('database/' + self.stdfile + '_mols.csv')
        self.dilutions = ['1:1', '1:5', '1:10', '1:50', '1:100', '1:1000']
        self.D = len(self.database)
        self.L = len(self.dilutions)
        self.T = len(self.transformations)
        self.transformations_map = {}
        self.transformations_colours = {}
        cmap = self._get_cmap(self.T)
        for t in range(self.T):
            trans = self.transformations[t]
            self.transformations_map[trans] = t
            self.transformations_colours[trans] = cmap(t)
        
        # lots of dictionaries to store intermediate results
        self.trans_results = {}
        self.trans_logged_results = {}
        self.trans_masses = {}
        self.dilution_logged_results = {}
        self.dilution_masses = {}
        self.dilution_colours = ['b', 'g', 'r', 'k', 'y', 'm']        
        self.trans_freq = Counter()
        
    def _get_cmap(self, N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        cmap = "Set1" # more colour maps here http://matplotlib.org/examples/color/colormaps_reference.html
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color    
        
    def populate_matrix(self):
    
        print "Loading everything for " + self.stdfile
        
        # populate trans_results and trans_masses matrices
        for t in range(len(self.transformations)):
    
            # filename of the previously stored results for each transformation t
            fname = './transformations_data/' + self.stdfile + '/' + str(t) + '.txt'
    
            # results is I by L, where I is the no. of molecules in database and 
            # L is the no. of dilution levels
            results = np.loadtxt(fname)
            logged = np.log2(results) # log base 2 to follow the spreadsheet
            logged[logged==-np.inf]=0 # replace -inf with 0        
    
            # load the masses too
            fname = './transformations_data/' + self.stdfile + '/' + str(t) + '.mass.txt'    
            masses = np.loadtxt(fname)

            self.trans_results[t] = results
            self.trans_logged_results[t] = logged
            self.trans_masses[t] = masses
    
        # count most common adducts
        for t in range(len(self.transformations)):
            results = self.trans_results[t]
            nnz = np.count_nonzero(results)
            self.trans_freq[self.transformations[t]] = nnz    
    
        # populate dilution_intenses and dilution_masses matrices
        for l in range(self.L):
            logged_results = np.zeros((self.D, self.T))
            masses = np.zeros((self.D, self.T))
            for t in range(self.T):
                temp = self.trans_logged_results[t]
                logged_results[:, t] = temp[:, l]
                temp = self.trans_masses[t]
                masses[:, t] = temp[:, l]
            self.dilution_logged_results[l] = logged_results
            self.dilution_masses[l] = masses           
            
        print "DONE"
        
    def make_heatmap(self):
        for t in range(len(self.transformations)):
            print self.transformations[t]
            logged_results = self.trans_logged_results[t]        
            sres = logged_results[logged_results[:, 0].argsort()] # sort by first column
            plt.pcolor(sres)
            plt.xlabel('dilutions')
            plt.ylabel('molDB')
            plt.title("Detected peaks log intensities (" + self.transformations[t].name + ")")
            plt.show()
    
    def print_most_common(self):
        for (key, val) in self.trans_freq.most_common():
            print key.name + " = " + str(val)
            
    def plot_adducts(self, n, show_mz=False):
    
        fig, axs = plt.subplots(1, self.L, sharex=False, sharey=True)
        
        for l in range(self.L):
            ax = axs[l]
            logged = self.dilution_logged_results[l]
            logged = logged[0:n, :]
            ax.matshow(logged, aspect='equal')
            ax.set_title(self.dilutions[l]) 
            if l == 0:
                ax.set_ylabel("molDB")
            ax.set_xlabel("trans")
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
        if show_mz:
            fig, axs = plt.subplots(1, self.L, sharex=False, sharey=True)
            fig.set_size_inches(20, 2)
            for l in range(self.L):
                ax = axs[l]
                c = self.dilution_colours[l]
                logged = self.dilution_logged_results[l]
                logged = logged[0:n, :]
                masses = self.dilution_masses[l]
                masses = masses[0:n, :]
                for (i,t), intensity in np.ndenumerate(logged):
                    mass = masses[i, t]
                    ax.plot((mass, mass), (0, intensity), c)
                ax.set_title(self.dilutions[l]) 
                if l == 0:
                    ax.set_ylabel("log intensity")
                ax.set_xlabel("m/z")
            plt.tight_layout()
            plt.show()
            
    def plot_compound(self, n):
    
        print "Top " + str(n) + " most common adducts"
        adducts = []
        for (key, val) in self.trans_freq.most_common(n):
            print key.name + " = " + str(val)
            adducts.append(key)
        print
        
        for i in range(self.D):
            db_entry = self.database[i]
            has_plot = False
            intenses_map = {}
            for trans in adducts:
                t = self.transformations_map[trans]
                all_ls = range(self.L)
                all_intenses = []
                for l in all_ls:
                    logged = self.dilution_logged_results[l]
                    intensity = logged[i, t]
                    all_intenses.append(intensity)
                if np.sum(all_intenses)>0:
                    plt.plot(all_ls, all_intenses, label=trans.name, 
                             color=self.transformations_colours[trans], linewidth=4)
                    has_plot = True
                    intenses_map[trans] = all_intenses
            if has_plot:
                print db_entry
                for trans in adducts:
                    if trans in intenses_map:
                        intensities = intenses_map[trans]
                        formatted_intensities = ['{:.3f}'.format(intensity) for intensity in intensities]
                        print "- " + trans.name + " = [" + ", ".join(formatted_intensities) + "]"                                           
                plt.title(db_entry.name + " (" + db_entry.formula + ")") 
                plt.ylabel("log intensity")
                plt.xlabel("dilutions")
                fontP = FontProperties()
                fontP.set_size('small')
                plt.legend(loc='upper right', prop = fontP)        
                plt.xticks(all_ls, self.dilutions)        
                plt.tight_layout()
                plt.show()
    
    def _get_cooccurence(self, t1, t2, l):
        t1mat = self.trans_results[t1]>0
        t2mat = self.trans_results[t2]>0
        t1mat = t1mat[:, l]
        t2mat = t2mat[:, l]
        t1mat = t1mat[:, None]
        t2mat = t2mat[:, None]
        results = np.dot(t1mat, t2mat.T)
        return results
    
    def plot_cooccurence(self, t1, t2):
        print "Co-occurence of compounds having " + str(self.transformations[t1].name) + " vs " + \
            str(self.transformations[t2].name + " adduct peaks in the same dilution file")
        for l in range(self.L):
            results = self._get_cooccurence(t1, t2, l)
            plt.spy(results, markersize=1)
            plt.ylabel("molDB (" + str(self.transformations[t1].name) + ")")
            plt.xlabel("molDB (" + str(self.transformations[t2].name) + ")")
            plt.title("Dilution " + self.dilutions[l])
            plt.show()