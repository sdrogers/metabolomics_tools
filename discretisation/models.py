from collections import namedtuple
import csv

import numpy as np
import scipy.sparse as sp


DatabaseEntry = namedtuple('DatabaseEntry', ['db_id', 'name', 'formula', 'mass'])
Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])

class HyperPars(object):

    def __init__(self):
        self.rt_prec = 100
        self.mass_prec = 100
        self.rt_prior_prec = 10
        self.mass_prior_prec = 10
        self.alpha = float(1)

    def __repr__(self):
        return "Hyperparameters: rt precision = " + str(self.rt_prec) + " mass precision = " + str(self.mass_prec) + \
            " rt prior precision = " + str(self.rt_prior_prec) + " mass prior precision = " + str(self.mass_prior_prec) + \
            " alpha = " + str(self.alpha)

class Feature(object):
            
    def __init__(self, feature_id, mass, rt, intensity):
        self.feature_id = feature_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.gt_metabolite = None # used for synthetic data
        self.gt_adduct = None # used for synthetic data
        
    def __repr__(self):
        return "Feature id=" + str(self.feature_id) + " mass=" + str(self.mass) + \
            " rt=" + str(self.rt) + " intensity=" + str(self.intensity) + \
            " gt_metabolite=" + str(self.gt_metabolite) + " gt_adduct=" + str(self.gt_adduct)

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
        self.possible, self.transformed, self.precursor_mass = self.discretise(mass_tol, rt_tol)
                
    def mass_match(self, mass, other_masses, tol):
        return np.abs((mass-other_masses)/mass)<tol*1e-6
    
    def rt_match(self, rt, other_rts, tol):
        return np.abs(rt-other_rts)<tol

    def discretise(self, mass_tol, rt_tol):       
        """ Discretise peaks by mass_tol and rt_tol, based on the occurence of possible precursor masses.

            Args: 
             - mass_tol: the mass tolerance for binning
             - rt_tol: the RT tolerance for binning

            Returns:
             - possible: an NxN matrix, where the entries are the index of the possible transformation from peak n to cluster k
             - transformed: an NxN matrix, where the entries are the actual transformed mass from peak n to cluster k             
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
        cluster_prior_mass_mean = (self.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos] 

        # N x N, entry is the id of mass_ok valid transformation originating from mass_ok peak
        possible = sp.lil_matrix((num_peaks, num_peaks), dtype=np.int)
        
        # N x N, entry is the possible precursor mass after applying the transformation
        transformed = sp.lil_matrix((num_peaks, num_peaks), dtype=np.float)
        
        # populate the matrices
        for n in np.arange(self.num_peaks):
            current_mass, current_rt, current_intensity = self.mass[n], self.rt[n], self.intensity[n]
            prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
            rt_ok = self.rt_match(current_rt, self.rt, rt_tol)
            intensity_ok = (current_intensity <= self.intensity)
            for t in np.arange(adduct_sub.size):
                mass_ok = self.mass_match(prior_mass[t], cluster_prior_mass_mean, mass_tol)
                pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                
        return possible, transformed, cluster_prior_mass_mean

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
                feature = Feature(feature_id=self.num(elements[0]), mass=self.num(elements[1]), \
                                  rt=self.num(elements[2]), intensity=self.num(elements[3]))
                features.append(feature)
        return features
    
    def load_features_sima(self, input_file):
        features = []
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                mass = self.num(elements[0])
                charge = self.num(elements[1]) # unused
                intensity = self.num(elements[2])
                rt = self.num(elements[3])
                gt_peak_id = self.num(elements[4]) # unused
                gt_metabolite_id = self.num(elements[5])
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
                                    mass=self.num(elements[3]))
                moldb.append(mol)
        return moldb
    
    def load_transformation(self, transformation):
        transformations = []
        with open(transformation, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 1
            for elements in reader:
                name=elements[0]
                sub=self.num(elements[1])
                mul=self.num(elements[2])
                iso=self.num(elements[3])
                trans = Transformation(i, name, sub, mul, iso)
                transformations.append(trans)
                i = i + 1
        return transformations        
    
    def num(self, s):
        try:
            return int(s)
        except ValueError:
            return float(s)

# Not sure whether want to keep this or not ...
class MassBin:
    def __init__(self, bin_id, start_mass, end_mass):
        self.bin_id = bin_id
        self.start_mass = start_mass
        self.end_mass = end_mass
        self.features = []
        self.molecules = set()
    def get_begin(self):
        return self.start_mass
    def get_end(self):
        return self.end_mass
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
        return 'MassBin id=' + str(self.bin_id) + ' mass=(' + str(self.start_mass) + \
            ', ' + str(self.end_mass) + ') num_features=' + str(len(self.features)) + \
            ' num_molecules=' + str(len(self.molecules))