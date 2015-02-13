from collections import namedtuple
import csv

import numpy as np
import scipy.sparse as sp
import utils
from matplotlib import pylab as plt


DatabaseEntry = namedtuple('DatabaseEntry', ['db_id', 'name', 'formula', 'mass'])
Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])


class ClusterPlotter(object):
    # an uncommented class for plotting clusters
    def __init__(self,peak_data,cluster_model):
        self.cluster_model = cluster_model
        self.peak_data = peak_data
        self.cluster_membership = (cluster_model.peak_cluster_probs>0.5)

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
                    str(1.0/self.cluster_model.cluster_rt_prec[cluster]) + ") Mean Mass: " + str(self.cluster_model.cluster_mass_mean[cluster]) + \
                    "(" + str(1.0/self.cluster_model.cluster_mass_prec[cluster]) + ")"
                plt.title(title_string) 

        
    def intensity_plot(self):
        # This will create the plot of intensity ratios versus intensity ratios
        print "hello"


class HyperPars(object):

    def __init__(self):
        
        self.rt_prec = 10
        self.mass_prec = 100
        self.rt_prior_prec = 100
        self.mass_prior_prec = 100
        self.alpha = float(100)

        self.discrete_alpha = 0.01
        self.discrete_rt_stdev = 20
        self.discrete_rt_prior_prec = 5E-3

    def __repr__(self):
        return "Hyperparameters " + utils.print_all_attributes(self)

class Feature(object):
            
    def __init__(self, feature_id, mass, rt, intensity):
        self.feature_id = feature_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.gt_metabolite = None   # used for synthetic data
        self.gt_adduct = None       # used for synthetic data
        
    def __repr__(self):
        return "Feature " + utils.print_all_attributes(self)

class PeakData(object):
    
    def __init__(self, features, database, transformations, mass_tol, rt_tol):
                
        # list of feature, database entry and transformation objects
        self.features = features
        self.database = database
        self.transformations = transformations
        self.num_peaks = len(features)

        # the same data as numpy arrays for convenience
        self.mass = np.array([f.mass for f in self.features])[:, None]              # N x 1 
        self.rt = np.array([f.rt for f in self.features])[:, None]                  # N x 1 
        self.intensity = np.array([f.intensity for f in self.features])[:, None]    # N x 1
        
        # discretise the input data
        self.possible, self.transformed, self.precursor_mass, self.matRT, self.bins = self.discretise(mass_tol, rt_tol)
                
    def mass_match(self, mass, other_masses, tol):
        return np.abs((mass-other_masses)/mass)<tol*1e-6
    
    def rt_match(self, rt, other_rts, tol):
        return np.abs(rt-other_rts)<tol
    
    def make_precursor_bin(self, bin_id, mass_centre, rt_centre, mass_tol, rt_tol):
        mass_centre = np.asscalar(mass_centre)
        rt_centre = np.asscalar(rt_centre)        
        # the intervals computed here should really be the same as whatever used in mass_match() and rt_match()
        interval = mass_centre * mass_tol * 1e-6
        mass_start = mass_centre - interval
        mass_end = mass_centre + interval
        rt_start = rt_centre - rt_tol
        rt_end = rt_centre + rt_tol
        pcb = PrecursorBin(bin_id, mass_start, mass_end, rt_start, rt_end)
        return pcb

    def discretise(self, mass_tol, rt_tol):       
        """ Discretise peaks by mass_tol and rt_tol, based on the occurence of possible precursor masses.

            Args: 
             - mass_tol: the mass tolerance for binning
             - rt_tol: the RT tolerance for binning

            Returns:
             - possible, transformed, bin_centre, mat_RT, bins
        """
        print 'Discretising peak data'
        adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1
        num_peaks = len(self.features)

        # find the location of M+H adduct in the transformation file
        proton_pos = np.flatnonzero(np.array(adduct_name)=='M+H') 

        # for each peak, calculate the prior precursor masses under M+H
        bin_centre = (self.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos] 

        # N x N, entry is the transformation id
        possible = sp.lil_matrix((num_peaks, num_peaks), dtype=np.int)
        
        # N x N, entry is the possible precursor mass after applying the transformation
        transformed = sp.lil_matrix((num_peaks, num_peaks), dtype=np.float)
        
        # N x N, entry is the retention time
        mat_RT = sp.lil_matrix((num_peaks,num_peaks), dtype = np.float)

        # populate the matrices
        bins = []
        for n in np.arange(self.num_peaks):

            current_mass, current_rt, current_intensity = self.mass[n], self.rt[n], self.intensity[n]
            pc_bin = self.make_precursor_bin(n, bin_centre[n], current_rt, mass_tol, rt_tol)
            bins.append(pc_bin)
            
            prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
            rt_ok = self.rt_match(current_rt, self.rt, rt_tol)
            intensity_ok = (current_intensity <= self.intensity)
            for t in np.arange(adduct_sub.size):
                mass_ok = self.mass_match(prior_mass[t], bin_centre, mass_tol)
                pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                mat_RT[n,pos] = current_rt
                
        return possible, transformed, bin_centre, mat_RT, bins

class FileLoader:
    
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol):
        """ Load everything that a clustering model requires """

        if input_file.endswith(".csv"):
            features = self.load_features(input_file)
        elif input_file.endswith(".txt"):
            # in SIMA (.txt) format, used for some old synthetic data
            features = self.load_features_sima(input_file)        
        
        # load database and transformations
        database = self.load_database(database_file)
        transformations = self.load_transformation(transformation_file)
        
        # discretisation happens inside PeakData
        data = PeakData(features, database, transformations, mass_tol, rt_tol)
        return data
        
    def load_features(self, input_file):
        features = []
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=':')
            next(reader, None)  # skip the headers
            for elements in reader:
                feature = Feature(feature_id=utils.num(elements[0]), mass=utils.num(elements[1]), \
                                  rt=utils.num(elements[2]), intensity=utils.num(elements[3]))
                features.append(feature)
        return features
    
    def load_features_sima(self, input_file):
        features = []
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                mass = utils.num(elements[0])
                charge = utils.num(elements[1]) # unused
                intensity = utils.num(elements[2])
                rt = utils.num(elements[3])
                gt_peak_id = utils.num(elements[4]) # unused
                gt_metabolite_id = utils.num(elements[5])
                gt_adduct_type = elements[6]
                feature = Feature(feature_id, mass, rt, intensity)
                feature.gt_metabolite = gt_metabolite_id
                feature.gt_adduct = gt_adduct_type
                features.append(feature)
                feature_id = feature_id + 1
        return features

    def load_database(self, database):
        moldb = []
        with open(database, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for elements in reader:
                mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                    mass=utils.num(elements[3]))
                moldb.append(mol)
        return moldb
    
    def load_transformation(self, transformation):
        transformations = []
        with open(transformation, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 1
            for elements in reader:
                name=elements[0]
                sub=utils.num(elements[1])
                mul=utils.num(elements[2])
                iso=utils.num(elements[3])
                trans = Transformation(i, name, sub, mul, iso)
                transformations.append(trans)
                i = i + 1
        return transformations        
    
# Not sure whether want to keep this or not ...
# Probably useful for identification and plotting later
class PrecursorBin(object):
    
    def __init__(self, bin_id, mass_start, mass_end, rt_start, rt_end):
        self.bin_id = bin_id
        self.mass_range = (mass_start, mass_end)
        self.rt_range = (rt_start, rt_end)
        self.features = []
        self.molecules = set()
        
    def get_begin(self):
        return self.mass_start

    def get_end(self):
        return self.mass_end
    
    def add_feature(self, feature):
        self.features.append(feature)
    
    def remove_feature(self, feature):
        if feature in self.features: 
            self.features.remove(feature)
    
    def get_features_count(self):
        return len(self.features)
    
    def get_features_rt(self):
        total_rt = 0
        for feature in self.features:
            total_rt = total_rt + feature.rt
        return total_rt
    
    def add_molecule(self, molecule):
        self.molecules.add(molecule)
    
    def __repr__(self):
        return "PrecursorBin " + utils.print_all_attributes(self)