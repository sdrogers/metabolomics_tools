from collections import namedtuple
import csv
import os

import numpy as np
import scipy.sparse as sp
import utils

Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])

class DatabaseEntry(object):
    # this is mostly called by FileLoader.load_database() 
    def __init__(self, db_id, name, formula, mass, mass_tol):
        self.db_id = db_id
        self.name = name
        self.formula = formula
        self.mass = mass
        self.mass_range = utils.mass_range(mass, mass_tol)
        
    def get_begin(self):
        return self.mass_range[0]

    def get_end(self):
        return self.mass_range[1]
        
    def __repr__(self):
        return "DatabaseEntry " + utils.print_all_attributes(self)

class HyperPars(object):

    def __init__(self):
        
        self.rt_prec = 10
        self.mass_prec = 100
        self.rt_prior_prec = 100
        self.mass_prior_prec = 100
        self.alpha = float(100)

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
    
    def __init__(self, features, database, transformations, mass_tol, rt_tol, make_bins=True):
                
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
        if make_bins:
            print 'Discretising peak data at mass_tol ' + str(mass_tol) + ' and rt_tol ' + str(rt_tol)
            self.possible, self.transformed, self.precursor_mass, self.matRT, self.bins = self.discretise(mass_tol, rt_tol)
                    
    def make_precursor_bin(self, bin_id, mass_centre, rt_centre, mass_tol, rt_tol):
        mass_centre = np.asscalar(mass_centre)
        rt_centre = np.asscalar(rt_centre)        
        pcb = PrecursorBin(bin_id, mass_centre, mass_tol, rt_centre, rt_tol)
        return pcb

    def discretise(self, mass_tol, rt_tol):       
        """ Discretise peaks by mass_tol and rt_tol, based on the occurence of possible precursor masses.

            Args: 
             - mass_tol: the mass tolerance for binning
             - rt_tol: the RT tolerance for binning

            Returns:
             - possible, transformed, bin_centre, mat_RT, bins
        """
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
            rt_ok = utils.rt_match(current_rt, self.rt, rt_tol)
            intensity_ok = (current_intensity <= self.intensity)
            for t in np.arange(len(self.transformations)):
                mass_ok = utils.mass_match(prior_mass[t], bin_centre, mass_tol)
                pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                mat_RT[n,pos] = current_rt
                
        print "{:d} bins created".format(len(bins))
        return possible, transformed, bin_centre, mat_RT, bins

class FileLoader:
    
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol, discretise=True):
        """ Load everything that a clustering model requires """

        if input_file.endswith(".csv"):
            features = self.load_features(input_file)
        elif input_file.endswith(".txt"):
            # in SIMA (.txt) format, used for some old synthetic data
            features = self.load_features_sima(input_file)        
        
        # load database and transformations
        database = self.load_database(database_file, mass_tol)
        transformations = self.load_transformation(transformation_file)
        
        # discretisation happens inside PeakData
        data = PeakData(features, database, transformations, mass_tol, rt_tol, make_bins=discretise)
        return data
        
    def load_features(self, input_file):
        features = []
        if not os.path.exists(input_file):
            return features
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
        if not os.path.exists(input_file):
            return features
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                mass = utils.num(elements[0])
                charge = utils.num(elements[1]) # unused
                intensity = utils.num(elements[2])
                rt = utils.num(elements[3])
                feature = Feature(feature_id, mass, rt, intensity)
                if len(elements)>4:
                    gt_peak_id = utils.num(elements[4]) # unused
                    gt_metabolite_id = utils.num(elements[5])
                    gt_adduct_type = elements[6]
                    feature.gt_metabolite = gt_metabolite_id
                    feature.gt_adduct = gt_adduct_type
                features.append(feature)
                feature_id = feature_id + 1
        return features

    def load_database(self, database, tol):
        moldb = []
        if not os.path.exists(database):
            return moldb
        with open(database, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for elements in reader:
                mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                    mass=utils.num(elements[3]), mass_tol=tol)
                moldb.append(mol)
        return moldb
    
    def load_transformation(self, transformation):
        transformations = []
        if not os.path.exists(transformation):
            return transformations
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
    
    def __init__(self, bin_id, mass_centre, mass_tol, rt_centre, rt_tol):
        self.bin_id = bin_id
        self.mass_centre = mass_centre
        self.rt_centre = rt_centre
        self.mass_range = utils.mass_range(mass_centre, mass_tol)
        self.rt_range = utils.rt_range(rt_centre, rt_tol)
        self.features = []
        self.molecules = set()
        
    def get_begin(self):
        return self.mass_range[0]

    def get_end(self):
        return self.mass_range[1]
    
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