import csv
import glob
import os
import sys

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

        self.adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        self.adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        self.adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        self.adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1

        # find index of M+H adduct in the list of transformations
        self.proton_pos = np.flatnonzero(np.array(self.adduct_name)=='M+H') 
        
        self.current_features = []
    
    def run_single(self, features):       

        # make bins using all the features in the file
        N = len(features)
        K = N # by definition
        feature_masses = np.array([f.mass for f in features])[:, None]              # N x 1
        prior_rts = np.array([f.rt for f in features])[:, None]                     # K x 1
        prior_intensities = np.array([f.intensity for f in features])[:, None]      # K x 1
        prior_masses = (feature_masses - self.adduct_sub[self.proton_pos])/self.adduct_mul[self.proton_pos] # K x 1
        matRT = sp.lil_matrix((N, K), dtype=np.float)                                   # N x K, RTs of feature n in bin k
        possible = sp.lil_matrix((N, K), dtype=np.int)                                  # N x K, transformation id+1 of feature n in bin k
        transformed = sp.lil_matrix((N, K), dtype=np.float)                             # N x K, transformed masses of feature n in bin k
        bins = []
        
        # populate possible, transformed, matRT
        for n in range(N):
            
            if n%100 == 0:
                sys.stdout.write('.')

            current_mass, current_rt, current_intensity = feature_masses[n], prior_rts[n], prior_intensities[n]
            pc_bin = self._make_precursor_bin(n, prior_masses[n], current_rt, current_intensity, self.mass_tol, self.rt_tol)
            bins.append(pc_bin)
            
            prior_mass = (current_mass - self.adduct_sub)/self.adduct_mul + self.adduct_del
            rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
            intensity_ok = (current_intensity <= prior_intensities)
            for t in np.arange(len(self.transformations)):
                mass_ok = utils.mass_match(prior_mass[t], prior_masses, self.mass_tol)
                check = rt_ok*mass_ok*intensity_ok
                # check = rt_ok*mass_ok                
                pos = np.flatnonzero(check)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                matRT[n, pos] = current_rt            

        print
        print "Total bins=" + str(K) + " total features=" + str(N)
        binning = DiscreteInfo(possible, transformed, matRT, bins, prior_masses, prior_rts)
        self.current_features.extend(features)
        return binning         

    def run_multiple(self, common, new_features):
                        
        # if there's some existing common bins, use them .. otherwise initialise an empty list
        if common is None:
            common_bins = []
        else:
            common_bins = common.bins

        # for each potential new cluster (i.e. feature) ..
        for n in range(len(new_features)):

            if n%200 == 0:
                sys.stdout.write('.')                             

            f = new_features[n]
            precursor_mass = (f.mass - self.adduct_sub[self.proton_pos])/self.adduct_mul[self.proton_pos]                 
            current_mass, current_rt, current_intensity = f.mass, f.rt, f.intensity         
               
            if len(common_bins) == 0:
                # if there's no bin yet, then f automatically becomes a bin
                bin_id = 0
                pc_bin = self._make_precursor_bin(bin_id, precursor_mass, f.rt, f.intensity, 
                                                  self.mass_tol, self.rt_tol)
                common_bins.append(pc_bin)                                                  
            else:
                # otherwise check whether this feature can go into any of the common bins
                prior_masses = np.array([b.mass for b in common_bins])[:, None]
                prior_rts = np.array([b.rt for b in common_bins])[:, None]
                prior_intensities = np.array([b.intensity for b in common_bins])[:, None]
                max_bin_id = np.array([b.bin_id for b in common_bins]).max()
                
                mass_ok = utils.mass_match(precursor_mass, prior_masses, self.mass_tol)
                rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
                intensity_ok = (current_intensity <= prior_intensities)
                check = self._check(mass_ok, rt_ok, intensity_ok)
                
                # if no common bin fits this feature
                count = check.sum()
                if count==0:
                    # then make new bin from the feature
                    max_bin_id += 1
                    pc_bin = self._make_precursor_bin(max_bin_id, precursor_mass, current_rt, current_intensity, 
                                                      self.mass_tol, self.rt_tol)
                    common_bins.append(pc_bin)

        self.current_features.extend(new_features)
        N = len(self.current_features)
        K = len(common_bins)

        # rebuild the matrices
        prior_masses = np.array([b.mass for b in common_bins])[:, None]            # K x 1
        prior_rts = np.array([b.rt for b in common_bins])[:, None]                 # K x 1
        prior_intensities = np.array([b.intensity for b in common_bins])[:, None]  # K x 1
        matRT = sp.lil_matrix((N, K), dtype=np.float)       # N x K, RTs of f n in bin k
        possible = sp.lil_matrix((N, K), dtype=np.int)      # N x K, transformation id+1 of f n in bin k
        transformed = sp.lil_matrix((N, K), dtype=np.float) # N x K, transformed masses of f n in bin k
        for n in range(N):
            
            if n%200 == 0:
                sys.stdout.write('.')                            

            f = self.current_features[n]    
            current_mass, current_rt, current_intensity = f.mass, f.rt, f.intensity
            
            transformed_masses = (current_mass - self.adduct_sub)/self.adduct_mul + self.adduct_del
            rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
            intensity_ok = (current_intensity <= prior_intensities)
            for t in np.arange(len(self.transformations)):
                mass_ok = utils.mass_match(transformed_masses[t], prior_masses, self.mass_tol)
                check = self._check(mass_ok, rt_ok, intensity_ok)            
                pos = np.flatnonzero(check)
                possible[n, pos] = t+1
                transformed[n, pos] = transformed_masses[t]
                matRT[n, pos] = current_rt            
                
        print
        print "Total clusters=" + str(K) + " total features=" + str(N)        
        new_binning = DiscreteInfo(possible, transformed, matRT, common_bins, prior_masses, prior_rts)
        return new_binning     
            
    def _make_precursor_bin(self, bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
        bin_mass = utils.as_scalar(bin_mass)
        bin_RT = utils.as_scalar(bin_RT)
        bin_intensity = utils.as_scalar(bin_intensity)
        pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
        return pcb
    
    def _check(self, mass_ok, rt_ok, intensity_ok):
        return mass_ok*rt_ok*intensity_ok

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
            
            # load the files
            file_id = 0
            data_list = []
            for file_path in filelist:
                features = self.load_features(file_path, synthetic=synthetic)
                for f in features:
                    f.file_id = file_id
                file_id += 1
                data = PeakData(features, database, transformations)
                data_list.append(data)

            # bin the files if necessary
            if make_bins:
                common = None
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                for j in range(len(data_list)):
                    print "Binning file " + str(j)
                    data = data_list[j]
                    features = data.features
                    common = discretiser.run_multiple(common, features) # and make common bins shared across files                   
                # assign common bins to all peak data
                for data in data_list:
                    data.set_discrete_info(common)
                
            return data_list
                    
        else:            
            # process only a single file
            features = self.load_features(input_file, synthetic=synthetic)
            binning = None
            if make_bins:
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                binning = discretiser.run_multiple(None, features)
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
        print str(len(features)) + " features read from " + input_file             
        return features
    
    def detect_delimiter(self, input_file):
        with open(input_file, 'rb') as csvfile:
            header = csvfile.readline()
            if header.find(":")!=-1:
                return ':'
            elif header.find(",")!=-1:
                return ','
    
    def load_features_csv(self, input_file):
        features = []
        if not os.path.exists(input_file):
            return features
        delim = self.detect_delimiter(input_file)
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=delim)
            next(reader, None)  # skip the headers
            for elements in reader:
                if len(elements)==6:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])                    
                    feature = Feature(feature_id=feature_id, mass=mz, rt=rt, intensity=0)                    
                    feature.into = utils.num(elements[3]) # integrated peak intensity
                    feature.maxo = utils.num(elements[4]) # maximum peak intensity
                    feature.intb = utils.num(elements[5]) # baseline corrected integrated peak intensities
                    feature.intensity = feature.maxo # we will use this for now
                elif len(elements)==5:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])                    
                    intensity = utils.num(elements[3])
                    identification = elements[4] # unused
                    feature = Feature(feature_id=feature_id, mass=mz, rt=rt, intensity=intensity)
                elif len(elements)==4:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])
                    intensity = utils.num(elements[3])
                    feature = Feature(feature_id=feature_id, mass=mz, rt=rt, intensity=intensity)
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