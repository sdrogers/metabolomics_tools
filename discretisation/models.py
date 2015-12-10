from collections import namedtuple

import numpy as np
import utils


class HyperPars(object):

    def __init__(self):
        self.rt_prec = 10
        self.mass_prec = 100
        self.rt_prior_prec = 100
        self.mass_prior_prec = 100
        self.alpha = float(100)

    def __repr__(self):
        return "Hyperparameters " + utils.print_all_attributes(self)

class DatabaseEntry(object):
    
    # this is mostly called by io.FileLoader.load_database() 
    def __init__(self, db_id, name, formula, mass, rt):
        self.db_id = db_id
        self.name = name
        self.formula = formula
        self.mass = mass
        self.rt = rt
        
    def set_ranges(self, mass_tol):
        self.mass_range = utils.mass_range(self.mass, mass_tol)
        
    def get_begin(self):
        return self.mass_range[0]

    def get_end(self):
        return self.mass_range[1]
    
    def __key(self):
        return (self.db_id, self.name)

    def __eq__(self, x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())       
        
    def __repr__(self):
        return "DatabaseEntry " + utils.print_all_attributes(self)

class Feature(object):
            
    def __init__(self, feature_id, mass, rt, intensity, file_id):
        self.feature_id = feature_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.file_id = file_id
        self.gt_metabolite = None   # used for synthetic data
        self.gt_adduct = None       # used for synthetic data
        
    def _get_key(self):
        return (self.feature_id, self.file_id)

    def __eq__(self, other):
        return self._get_key() == other._get_key()

    def __hash__(self):
        return hash(self._get_key())    
        
    def __repr__(self):
        return "id=(%d,%d) mass=%.4f rt=%.2f int=%.2f" % (self.feature_id, self.file_id, self.mass, self.rt, self.intensity)

Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])
DiscreteInfo = namedtuple('DiscreteInfo', ['possible', 'transformed', 'matRT', 'bins', 'prior_masses', 'prior_rts'])
    
class PeakData(object):
    
    def __init__(self, features, filename, corr_mat=None):
                
        # list of feature, database entry and transformation objects
        self.features = features
        self.num_peaks = len(features)
        self.corr_mat = corr_mat
        self.filename = filename

        # the same data as numpy arrays for convenience
        self.mass = np.array([f.mass for f in self.features])[:, None]              # N x 1 
        self.rt = np.array([f.rt for f in self.features])[:, None]                  # N x 1 
        self.intensity = np.array([f.intensity for f in self.features])[:, None]    # N x 1
                
# Not sure whether want to keep this or not ...
# Probably useful for identification and plotting later
class PrecursorBin(object):
    
    def __init__(self, bin_id, mass, rt, intensity, mass_tol, rt_tol):
        self.bin_id = bin_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.mass_range = utils.mass_range(mass, mass_tol)
        self.rt_range = utils.rt_range(rt, rt_tol)
        self.features = set()
        self.molecules = set()
        self.origin = 0
        
    def get_begin(self):
        return self.mass_range[0]

    def get_end(self):
        return self.mass_range[1]
    
    def add_feature(self, feature):
        self.features.add(feature)
    
    def remove_feature(self, feature):
        self.features.remove(feature)
    
    def get_features_count(self):
        return len(self.features)
    
    def get_features_rt(self):
        total_rt = 0
        for feature in self.features:
            if isinstance(feature, tuple):
                # if it's a list of tuples, then take the first element
                total_rt += feature[0].rt
            else:
                total_rt += feature.rt
        return total_rt
    
    def add_molecule(self, molecule):
        self.molecules.add(molecule)

    def remove_molecule(self, molecule):
        self.molecules.remove(molecule)
        
    def __key(self):
        return (self.bin_id, self.origin)

    def __eq__(self, x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())           
    
    def __repr__(self):
        return "PrecursorBin " + utils.print_all_attributes(self)