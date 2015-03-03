import csv
import glob
import os

from models import DiscreteInfo, PrecursorBin, PeakData, Feature, DatabaseEntry, Transformation
import numpy as np
import scipy.sparse as sp
import utils


class Discretiser(object):

    def __init__(self, transformations, mass_tol, rt_tol):

        print "Discretising at mass_tol=" + str(mass_tol) + ", rt_tol=" + str(rt_tol)
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
            feature_masses = np.array([f.mass for f in new_features])[:, None]              # N x 1
            prior_rts = np.array([f.rt for f in new_features])[:, None]                     # N x 1
            prior_intensities = np.array([f.intensity for f in new_features])[:, None]      # N x 1

            prior_masses = (feature_masses - adduct_sub[proton_pos])/adduct_mul[proton_pos] # K x 1
            matRT = sp.lil_matrix((N, K), dtype=np.float)                                   # N x K, RTs of feature n in bin k
            possible = sp.lil_matrix((N, K), dtype=np.int)                                  # N x K, transformation id+1 of feature n in bin k
            transformed = sp.lil_matrix((N, K), dtype=np.float)                             # N x K, transformed masses of feature n in bin k
            bins = []
            
            # populate possible, transformed, matRT
            for n in range(N):
                
                if n%100 == 0:
                    print '.',
    
                current_mass, current_rt, current_intensity = feature_masses[n], prior_rts[n], prior_intensities[n]
                pc_bin = self._make_precursor_bin(n, prior_masses[n], current_rt, current_intensity, self.mass_tol, self.rt_tol)
                bins.append(pc_bin)
                
                prior_mass = (current_mass - adduct_sub)/adduct_mul + adduct_del
                rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
                intensity_ok = (current_intensity <= prior_intensities)
                for t in np.arange(len(self.transformations)):
                    mass_ok = utils.mass_match(prior_mass[t], prior_masses, self.mass_tol)
                    pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                    possible[n, pos] = t+1
                    transformed[n, pos] = prior_mass[t]
                    matRT[n, pos] = current_rt            

            print
            print "Total bins=" + str(K) + " total features=" + str(N)
            binning = DiscreteInfo(possible, transformed, matRT, bins, prior_masses, prior_rts)
            return binning         
        
        else: # otherwise need to do some checking ...
        
            # we want to make new bins only for features that cannot go into any existing bins from the previous files
            bins = binning.bins
            prior_masses = np.array([b.mass for b in bins])[:, None]
            prior_rts = np.array([b.rt for b in bins])[:, None]
            prior_intensities = np.array([b.intensity for b in bins])[:, None]
            max_bin_id = np.array([b.bin_id for b in bins]).max()
            
            # for each potential new cluster ..
            for can in new_features:
                
                precursor_mass = (can.mass - adduct_sub[proton_pos])/adduct_mul[proton_pos] 
                mass_ok = utils.mass_match(precursor_mass, prior_masses, self.mass_tol)
                rt_ok = utils.rt_match(can.rt, prior_rts, self.rt_tol)
                intensity_ok = (can.intensity <= prior_intensities)
                check = rt_ok*mass_ok*intensity_ok
                
                # if no suitable existing bin found
                if check.sum(0)==0:
                    # then make new bin
                    max_bin_id += 1
                    pc_bin = self._make_precursor_bin(max_bin_id, precursor_mass, can.rt, can.intensity, 
                                                      self.mass_tol, self.rt_tol)
                    bins.append(pc_bin)

            # rebuild the matrices
            prior_masses = np.array([b.mass for b in bins])[:, None]            # K x 1
            prior_rts = np.array([b.rt for b in bins])[:, None]                 # K x 1
            prior_intensities = np.array([b.intensity for b in bins])[:, None]  # K x 1

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
                rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
                intensity_ok = (current_intensity <= prior_intensities)
                for t in np.arange(len(self.transformations)):
                    mass_ok = utils.mass_match(prior_mass[t], prior_masses, self.mass_tol)
                    pos = np.flatnonzero(rt_ok*mass_ok*intensity_ok)
                    possible[n, pos] = t+1
                    transformed[n, pos] = prior_mass[t]
                    matRT[n, pos] = current_rt            
                            
            print "Total bins=" + str(K) + " total features=" + str(N)
            binning = DiscreteInfo(possible, transformed, matRT, bins, prior_masses, prior_rts)
            return binning         
    
    def _make_precursor_bin(self, bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
        bin_mass = utils.as_scalar(bin_mass)
        bin_RT = utils.as_scalar(bin_RT)
        bin_intensity = utils.as_scalar(bin_intensity)
        pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
        return pcb

class FileLoader:
        
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol, 
                         make_bins=True, synthetic=False):
        """ Load everything that a clustering model requires """

        # load database and transformations
        database = self.load_database(database_file)
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
                new_features = self.load_features(file_path, synthetic=synthetic)
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
            features = self.load_features(input_file, synthetic=synthetic)
            binning = None
            if make_bins:
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                binning = discretiser.run([], features, None)
            data = PeakData(features, database, transformations, binning)
            return data
        
    def load_features(self, input_file, synthetic=False):
        features = []
        if input_file.endswith(".csv"):
            features = self.load_features_csv(input_file)
        elif input_file.endswith(".txt"):
            if synthetic:
                # in SIMA (.txt) format, used for some old synthetic data
                features = self.load_features_sima(input_file)
            else:
                # in tab-separated format from mzMatch
                features = self.load_features_txt(input_file)   
        print str(len(features)) + " features read"             
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

    def load_features_txt(self, input_file):
        features = []
        if not os.path.exists(input_file):
            return features
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                feature = Feature(feature_id=feature_id, mass=utils.num(elements[0]), \
                                  rt=utils.num(elements[1]), intensity=utils.num(elements[2]))
                features.append(feature)
                feature_id = feature_id + 1
        return features
        
    def load_features_sima(self, input_file):
        features = []
        if not os.path.exists(input_file):
            return features
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            feature_id = 1
            for elements in reader:
                mass = float(elements[0])
                charge = float(elements[1])
                mass = mass/charge
                intensity = utils.num(elements[2])
                rt = utils.num(elements[3])
                feature = Feature(feature_id, mass, rt, intensity)
                if len(elements)>4:
                    # for debugging with synthetic data
                    gt_peak_id = utils.num(elements[4])
                    gt_metabolite_id = utils.num(elements[5])
                    gt_adduct_type = elements[6]
                    feature.gt_metabolite = gt_metabolite_id
                    feature.gt_adduct = gt_adduct_type
                features.append(feature)
                feature_id = feature_id + 1
        return features

    def load_database(self, database):
        moldb = []
        if not os.path.exists(database):
            return moldb
        with open(database, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for elements in reader:
                if len(elements)==5:
                    mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                        mass=utils.num(elements[3]), rt=utils.num(elements[4]))
                    moldb.append(mol)
                elif len(elements)==4:
                    mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                        mass=utils.num(elements[3]), rt=0)                    
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