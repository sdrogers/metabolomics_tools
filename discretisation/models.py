from collections import namedtuple
import csv
import os
import glob

import numpy as np
import scipy.sparse as sp
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

class Feature(object):
            
    def __init__(self, feature_id, mass, rt, intensity, file_id=0):
        self.feature_id = feature_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.file_id = file_id
        self.gt_metabolite = None   # used for synthetic data
        self.gt_adduct = None       # used for synthetic data
        
    def __repr__(self):
        return "Feature " + utils.print_all_attributes(self)

Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])
DiscreteInfo = namedtuple('DiscreteInfo', ['possible', 'transformed', 'precursor_masses', 'precursor_rts', 'bins'])
    
class Discretiser(object):

    def __init__(self, transformations, mass_tol, rt_tol):

        print "Discretising peak data mass_tol=" + str(mass_tol) + ", rt_tol=" + str(rt_tol)
        self.transformations = transformations
        self.mass_tol = mass_tol
        self.rt_tol = rt_tol
    
    def run(self, features):       

        num_peaks = len(features)
        mass = np.array([f.mass for f in features])[:, None]                        # N x 1 
        rt = np.array([f.rt for f in features])[:, None]                            # N x 1 
        intensity = np.array([f.intensity for f in features])[:, None]              # N x 1
        adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1

        # find the location of M+H adduct in the transformation file
        proton_pos = np.flatnonzero(np.array(adduct_name)=='M+H') 

        # for each peak, calculate the bin's precursor masses using M+H transformation
        precursor_masses = (mass - adduct_sub[proton_pos])/adduct_mul[proton_pos] 

        # and also the bin's retention time
        precursor_rts = sp.lil_matrix((num_peaks,num_peaks), dtype = np.float)

        # N x N, entry is the transformation id
        possible = sp.lil_matrix((num_peaks, num_peaks), dtype=np.int)
        
        # N x N, entry is the transformed mass of a peak feature
        transformed = sp.lil_matrix((num_peaks, num_peaks), dtype=np.float)
        
        # populate the matrices
        bins = []
        for n in np.arange(num_peaks):

            current_mass, current_rt, current_intensity = mass[n], rt[n], intensity[n]
            pc_bin = self.__make_precursor_bin__(n, precursor_masses[n], current_rt, self.mass_tol, self.rt_tol)
            bins.append(pc_bin)
            
            prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
            rt_ok = utils.rt_match(current_rt, rt, self.rt_tol)
            intensity_ok = (current_intensity <= intensity)
            for t in np.arange(len(self.transformations)):
                mass_ok = utils.mass_match(prior_mass[t], precursor_masses, self.mass_tol)
                pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                precursor_rts[n, pos] = current_rt
                
        print "{:d} bins created".format(len(bins))
        discrete_info = DiscreteInfo(possible, transformed, precursor_masses, precursor_rts, bins)
        return discrete_info
    
    def __make_precursor_bin__(self, bin_id, mass_centre, rt_centre, mass_tol, rt_tol):
        mass_centre = np.asscalar(mass_centre)
        rt_centre = np.asscalar(rt_centre)        
        pcb = PrecursorBin(bin_id, mass_centre, mass_tol, rt_centre, rt_tol)
        return pcb

class PeakData(object):
    
    def __init__(self, features, database, transformations, discrete_info=None):
                
        # list of feature, database entry and transformation objects
        self.features = features
        self.database = database
        self.transformations = transformations
        self.num_peaks = len(features)

        # the same data as numpy arrays for convenience
        self.mass = np.array([f.mass for f in self.features])[:, None]              # N x 1 
        self.rt = np.array([f.rt for f in self.features])[:, None]                  # N x 1 
        self.intensity = np.array([f.intensity for f in self.features])[:, None]    # N x 1
        
        if discrete_info is not None:
            self.possible = discrete_info.possible
            self.transformed = discrete_info.transformed
            self.precursor_masses = discrete_info.precursor_masses
            self.precursor_rts = discrete_info.precursor_rts
            self.bins = discrete_info.bins
                            
class FileLoader:
        
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol, make_bins=True):
        """ Load everything that a clustering model requires """

        # load database and transformations
        database = self.load_database(database_file, mass_tol)
        transformations = self.load_transformation(transformation_file)
        if os.path.isdir(input_file):

            input_dir = input_file
            
            # find all .csv and .txt files inside input_dir
            filelist = []
            types = ('*.csv', '*.txt')
            os.chdir(input_dir)
            for files in types:
                filelist.extend(glob.glob(files))
            filelist = utils.natural_sort(filelist)
            
            # process each input file
            reference_bins = []        
            for fileitem in filelist:
                full_path = os.path.join(input_dir, fileitem);
                # handle file
                if input_file.endswith(".csv"):
                    features = self.load_features(full_path)
                elif input_file.endswith(".txt"):
                    # in SIMA (.txt) format, used for some old synthetic data
                    features = self.load_features_sima(full_path)                
                    
        else:
            
            # handle single file
            if input_file.endswith(".csv"):
                features = self.load_features(input_file)
            elif input_file.endswith(".txt"):
                # in SIMA (.txt) format, used for some old synthetic data
                features = self.load_features_sima(input_file)                
            
            # discretise the input data if necessary
            dc_res = None
            if make_bins:
                dc = Discretiser(transformations, mass_tol, rt_tol)
                dc_res = dc.run(features)
            data = PeakData(features, database, transformations, dc_res)
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