import csv
import glob
from operator import attrgetter
import os
import sys

from interval_tree import IntervalTree
from models import DiscreteInfo, PrecursorBin, PeakData, Feature, DatabaseEntry, Transformation
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import utils


class Discretiser(object):

    def __init__(self, transformations, mass_tol, rt_tol):

        self.transformations = transformations
        self.mass_tol = mass_tol
        self.rt_tol = rt_tol

        self.adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        self.adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        self.adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        self.adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1

        # find index of M+H adduct in the list of transformations
        self.proton_pos = np.flatnonzero(np.array(self.adduct_name)=='M+H') 
            
    def run_single(self, features):       

        print "Discretising at mass_tol=" + str(self.mass_tol) + ", rt_tol=" + str(self.rt_tol)
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
        return binning         

    def run_multiple(self, data_list):

        print "Discretising at mass_tol=" + str(self.mass_tol)
        sys.stdout.write('Making top-level bins shared across files ')
        all_features = []
        for peak_data in data_list:
            all_features.extend(peak_data.features)    
        all_features = sorted(all_features, key = attrgetter('mass'))            
        N = len(all_features)        
                        
        # define a set of top mass bins shared across files
        top_bin_masses = []
        for n in range(len(all_features)):

            if n%200 == 0:
                sys.stdout.write('.')                             

            f = all_features[n]
            precursor_mass = (f.mass - self.adduct_sub[self.proton_pos])/self.adduct_mul[self.proton_pos]        

            # check if any existing bin can fit this feature within some mass tolerance
            prior_masses = np.array(top_bin_masses)[:, None]
            mass_ok = utils.mass_match(precursor_mass, prior_masses, self.mass_tol)
                
            # if no common bin fits this feature
            count = mass_ok.sum()
            if count==0:
                # then make new bin from the feature
                top_bin_masses.append(np.asscalar(precursor_mass))

        K = len(top_bin_masses)
        top_bins = []
        for k in range(K):
            mass = top_bin_masses[k]
            pc_bin = self._make_precursor_bin(k, mass, 0, 0, self.mass_tol, 0)
            top_bins.append(pc_bin)            
        print
        print "Total top bins=" + str(K) + " total features=" + str(N)

        # for each file, we want to instantiatie its concrete bins -- based on the top bins
        all_binning = []
        for j in range(len(data_list)):
            
            sys.stdout.write("Instantiating concrete bins for file " + str(j) + " ")
            
            peak_data = data_list[j]            
            features = peak_data.features
            N = len(features)        

            # initialise the 'concrete' realisations of the top bins in this file
            concrete_bins = []
            k = 0
            for a in range(len(top_bins)):
                
                if a%200 == 0:
                    sys.stdout.write('.')                             
                
                # find all features that can fit by mass in the top level bin
                tb = top_bins[a]
                fs = self._find_features(tb, features) 
                if fs is not None:
                    for f1 in fs:
                        # make a new concrete bin from the feature based on mass and RT
                        precursor_mass = (f1.mass - self.adduct_sub[self.proton_pos])/self.adduct_mul[self.proton_pos] 
                        precursor_mass = np.asscalar(precursor_mass)
                        concrete_bin = PrecursorBin(k, precursor_mass, f1.rt, f1.intensity, self.mass_tol, self.rt_tol)
                        concrete_bin.top_id = tb.bin_id
                        concrete_bin.origin = j
                        concrete_bins.append(concrete_bin)
                        k += 1

            K = len(concrete_bins)
            print
            print "File " + str(j) + " has " + str(K) + " concrete bins instantiated"
            prior_masses = np.array([bb.mass for bb in concrete_bins])[:, None]                # K x 1                                
            prior_rts = np.array([bb.rt for bb in concrete_bins])[:, None]                     # K x 1
            prior_intensities = np.array([bb.intensity for bb in concrete_bins])[:, None]      # K x 1

            # build the matrices for this file
            matRT = sp.lil_matrix((N, K), dtype=np.float)       # N x K, RTs of f n in bin k
            possible = sp.lil_matrix((N, K), dtype=np.int)      # N x K, transformation id+1 of f n in bin k
            transformed = sp.lil_matrix((N, K), dtype=np.float) # N x K, transformed masses of f n in bin k
            sys.stdout.write("Building matrices for file " + str(j) + " ")
            for n in range(N):
                
                if n%200 == 0:
                    sys.stdout.write('.')                            
    
                f = features[n]    
                current_mass, current_rt, current_intensity = f.mass, f.rt, f.intensity
                transformed_masses = (current_mass - self.adduct_sub)/self.adduct_mul + self.adduct_del

                rt_ok = utils.rt_match(current_rt, prior_rts, self.rt_tol)
                intensity_ok = (current_intensity <= prior_intensities)
                for t in np.arange(len(self.transformations)):
                    # fill up the target bins that this transformation allows
                    mass_ok = utils.mass_match(transformed_masses[t], prior_masses, self.mass_tol)
                    check = mass_ok*rt_ok*intensity_ok
                    pos = np.flatnonzero(check)
                    # print (f.feature_id, t, pos)
                    possible[n, pos] = t+1
                    # and other prior values too
                    transformed[n, pos] = transformed_masses[t]
                    matRT[n, pos] = current_rt            
                    
                possible_clusters = np.nonzero(possible[n, :])[1]
                assert len(possible_clusters) > 0, str(f) + " has no possible clusters"
                
            print
            binning = DiscreteInfo(possible, transformed, matRT, concrete_bins, prior_masses, prior_rts)
            all_binning.append(binning)

        return all_binning     
            
    def _make_precursor_bin(self, bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol):
        bin_mass = utils.as_scalar(bin_mass)
        bin_RT = utils.as_scalar(bin_RT)
        bin_intensity = utils.as_scalar(bin_intensity)
        pcb = PrecursorBin(bin_id, bin_mass, bin_RT, bin_intensity, mass_tol, rt_tol)
        return pcb
    
    def _find_features(self, bb, features):
        masses = np.array([f.mass for f in features])
        precursor_masses = (masses - self.adduct_sub[self.proton_pos])/self.adduct_mul[self.proton_pos]
        check1 = bb.get_begin() < precursor_masses
        check2 = precursor_masses < bb.get_end()
        pos = np.flatnonzero(check1*check2)
        pos = pos.tolist()
        results = [features[i] for i in pos]
        if len(results) == 0:
            return None
        else:
            return results
    
class FileLoader:
        
    def load_model_input(self, input_file, database_file, transformation_file, mass_tol, rt_tol, 
                         make_bins=True, synthetic=False, limit_n=-1):
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
            starting_dir = os.getcwd() # save the initial dir to restore
            os.chdir(input_dir)
            for files in types:
                filelist.extend(glob.glob(files))
            filelist = utils.natural_sort(filelist)
            self.file_list = filelist
            
            # load the files
            file_id = 0
            data_list = []
            all_features = []
            for file_path in filelist:
                features, corr_mat = self.load_features(file_path, synthetic=synthetic)
                # TODO: feature really should be immutable!!
                for f in features:
                    f.file_id = file_id
                file_id += 1
                if limit_n > -1:
                    print "Using only " + str(limit_n) + " features from " + file_path
                    features = features[0:limit_n]
                data = PeakData(features, database, transformations, corr_mat=corr_mat)
                all_features.extend(features)
                data_list.append(data)
                
            # bin the files if necessary
            if make_bins:
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                # make common bins shared across files using all the features                   
                discrete_infos = discretiser.run_multiple(data_list) 
                assert len(data_list) == len(discrete_infos)
                for j in range(len(data_list)):
                    peak_data = data_list[j]
                    common = discrete_infos[j]
                    peak_data.set_discrete_info(common)
                            
            os.chdir(starting_dir)    
            return data_list
                    
        else:   
                     
            # process only a single file
            features, corr_mat = self.load_features(input_file, synthetic=synthetic)
            if limit_n > -1:
                features = features[0:limit_n]
            binning = None
            if make_bins:
                discretiser = Discretiser(transformations, mass_tol, rt_tol)
                binning = discretiser.run_single(features)
            data = PeakData(features, database, transformations, discrete_info=binning, corr_mat=corr_mat)
            return data
                
    def load_features(self, input_file, load_correlations=False, synthetic=False):

        # first load the features
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

        # also check if the correlation matrix is there, if yes load it too
        corr_mat = None
        if load_correlations:
            front_part, extension = os.path.splitext(input_file)
            matfile = front_part + '.corr.mat'
            if os.path.isfile(matfile):
                print "Reading peak shape correlations from " + matfile
                mdict = sio.loadmat(matfile)
                corr_mat = mdict['corr_mat'] 

        return features, corr_mat
    
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
