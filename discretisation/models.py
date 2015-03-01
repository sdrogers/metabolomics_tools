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
DiscreteInfo = namedtuple('DiscreteInfo', ['possible', 'transformed', 'precursor_masses', 'matRT', 'bins'])
    
class Discretiser(object):

    def __init__(self, transformations, mass_tol, rt_tol):

        print "\tDiscretising peak data mass_tol=" + str(mass_tol) + ", rt_tol=" + str(rt_tol)
        self.transformations = transformations
        self.mass_tol = mass_tol
        self.rt_tol = rt_tol
    
    def run(self, current_features, new_features, binning):       

        ### TODO: lots of code duplication in this method ..

        adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1

        # find index of M+H adduct in the list of transformations
        proton_pos = np.flatnonzero(np.array(adduct_name)=='M+H') 

        # if no binning information, i.e. first file
        if binning is None:

            # then make bins using all the features in the file
            N = len(new_features)
            K = N
            feature_masses = np.array([f.mass for f in new_features])[:, None]                      # N x 1
            bin_rts = np.array([f.rt for f in new_features])[:, None]                               # N x 1
            bin_intensities = np.array([f.intensity for f in new_features])[:, None]                # N x 1

            precursor_masses = (feature_masses - adduct_sub[proton_pos])/adduct_mul[proton_pos]     # K x 1
            matRT = sp.lil_matrix((N, K), dtype=np.float)                                           # N x K, RTs of feature n in bin k
            possible = sp.lil_matrix((N, K), dtype=np.int)                                          # N x K, transformation id+1 of feature n in bin k
            transformed = sp.lil_matrix((N, K), dtype=np.float)                                     # N x K, transformed masses of feature n in bin k
            bins = []
            
            # populate possible, transformed, matRT
            for n in range(N):
    
                current_mass, current_rt, current_intensity = feature_masses[n], bin_rts[n], bin_intensities[n]
                pc_bin = self._make_precursor_bin(n, precursor_masses[n], current_rt, current_intensity, self.mass_tol, self.rt_tol)
                bins.append(pc_bin)
                
                prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
                rt_ok = utils.rt_match(current_rt, bin_rts, self.rt_tol)
                intensity_ok = (current_intensity <= bin_intensities)
                for t in np.arange(len(self.transformations)):
                    mass_ok = utils.mass_match(prior_mass[t], precursor_masses, self.mass_tol)
                    pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                    possible[n, pos] = t+1
                    transformed[n, pos] = prior_mass[t]
                    matRT[n, pos] = current_rt            

            print "\t{:d} bins total".format(len(bins))
            binning = DiscreteInfo(possible, transformed, precursor_masses, matRT, bins)
            return binning         
        
        else: # otherwise need to do some checking ...
        
            # make new bins only for new features that cannot go into any existing bins
            bins = binning.bins
            precursor_masses = np.array([b.mass for b in bins])[:, None]
            bin_rts = np.array([b.rt for b in bins])[:, None]
            bin_intensities = np.array([b.intensity for b in bins])[:, None]
            max_bin_id = np.array([b.bin_id for b in bins]).max()
            
            for can in new_features:
                precursor_mass = (can.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos] 
                mass_ok = utils.mass_match(precursor_mass, precursor_masses, self.mass_tol)
                rt_ok = utils.rt_match(can.rt, bin_rts, self.rt_tol)
                intensity_ok = (can.intensity <= bin_intensities)
                check = rt_ok*mass_ok*intensity_ok
                if check.sum(0)==0: # if no suitable existing bin found
                    max_bin_id += 1
                    pc_bin = self._make_precursor_bin(max_bin_id, precursor_mass, can.rt, can.intensity, 
                                                      self.mass_tol, self.rt_tol)
                    bins.append(pc_bin)

            # rebuild the matrices
            precursor_masses = np.array([b.mass for b in bins])[:, None]        # K x 1
            bin_rts = np.array([b.rt for b in bins])[:, None]                   # K x 1
            bin_intensities = np.array([b.intensity for b in bins])[:, None]    # K x 1

            features = list(current_features)
            features.extend(new_features)
            N = len(features)
            K = len(bins)

            matRT = sp.lil_matrix((N, K), dtype=np.float)       # N x K, RTs of feature n in bin k
            possible = sp.lil_matrix((N, K), dtype=np.int)      # N x K, transformation id+1 of feature n in bin k
            transformed = sp.lil_matrix((N, K), dtype=np.float) # N x K, transformed masses of feature n in bin k

            # populate possible, transformed, matRT
            for n in range(N):
    
                f = features[n]    
                current_mass, current_rt, current_intensity = f.mass, f.rt, f.intensity
                
                prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
                rt_ok = utils.rt_match(current_rt, bin_rts, self.rt_tol)
                intensity_ok = (current_intensity <= bin_intensities)
                for t in np.arange(len(self.transformations)):
                    mass_ok = utils.mass_match(prior_mass[t], precursor_masses, self.mass_tol)
                    pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                    possible[n, pos] = t+1
                    transformed[n, pos] = prior_mass[t]
                    matRT[n, pos] = current_rt            
                            
            print "\t{:d} bins total".format(len(bins))
            binning = DiscreteInfo(possible, transformed, precursor_masses, matRT, bins)
            return binning         
    
    def _make_precursor_bin(self, bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
        bin_mass = utils.as_scalar(bin_mass)
        bin_RT = utils.as_scalar(bin_RT)
        bin_intensity = utils.as_scalar(bin_intensity)
        pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
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
            self.matRT = discrete_info.matRT
            self.bins = discrete_info.bins
            self.num_clusters = len(self.bins)
            
class FileLoader:
        
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol, make_bins=True):
        """ Load everything that a clustering model requires """

        # load database and transformations
        database = self.load_database(database_file, mass_tol)
        transformations = self.load_transformation(transformation_file)

        # if this is a directory, process all files inside
        if os.path.isdir(input_file):

            # find all the .txt and csv files in input_dir
            input_dir = input_file
            filelist = []
            types = ('*.csv', '*.txt')
            os.chdir(input_dir)
            for files in types:
                filelist.extend(glob.glob(files))
            filelist = utils.natural_sort(filelist)
            
            # process file one by one
            features = []
            binning = None
            file_id = 0
            for file_path in filelist:
                # file_path = os.path.abspath(file_path)
                new_features = self.load_features(file_path)
                for f in new_features:
                    f.file_id = file_id
                print "Processing file_id=" + str(file_id) + " " + file_path + " " + str(len(new_features)) + " features"
                file_id += 1
                if make_bins:
                    discretiser = Discretiser(transformations, mass_tol, rt_tol)
                    binning = discretiser.run(features, new_features, binning)
                features.extend(new_features)
            data = PeakData(features, database, transformations, binning)
            return data
                    
        else:            
            # process only a single file
            features = self.load_features(input_file)
            if make_bins:
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                binning = discretiser.run([], features, None)
            data = PeakData(features, database, transformations, binning)
            return data
        
    def load_features(self, input_file):
        features = []
        if input_file.endswith(".csv"):
            features = self.load_features_csv(input_file)
        elif input_file.endswith(".txt"):
            # in SIMA (.txt) format, used for some old synthetic data
            features = self.load_features_sima(input_file)                
        return features
    
    def load_features_csv(self, input_file):
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
    
    def __init__(self, bin_id, mass, rt, intensity, mass_tol, rt_tol):
        self.bin_id = bin_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.mass_range = utils.mass_range(mass, mass_tol)
        self.rt_range = utils.rt_range(rt, rt_tol)
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