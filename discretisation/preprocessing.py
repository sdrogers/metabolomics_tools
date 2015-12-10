import csv
import glob
import os
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from models import DiscreteInfo, PeakData, Feature, DatabaseEntry, Transformation
import utils

class Discretiser(object):

    def __init__(self, transformations, within_file_mass_tol, within_file_rt_tol, across_file_mass_tol, verbose=False):

        self.transformations = transformations
        self.within_file_mass_tol = within_file_mass_tol
        self.within_file_rt_tol = within_file_rt_tol
        self.across_file_mass_tol = across_file_mass_tol

        self.adduct_name = np.array([t.name for t in self.transformations])[:,None]      # A x 1
        self.adduct_mul = np.array([t.mul for t in self.transformations])[:,None]        # A x 1
        self.adduct_sub = np.array([t.sub for t in self.transformations])[:,None]        # A x 1
        self.adduct_del = np.array([t.iso for t in self.transformations])[:,None]        # A x 1

        # find index of M+H adduct in the list of transformations
        self.proton_pos = np.flatnonzero(np.array(self.adduct_name)=='M+H') 
        self.verbose = verbose
            
    def run_single(self, features):       

        print "Discretising at within_file_mass_tol=" + str(self.within_file_mass_tol) + ", within_file_rt_tol=" + str(self.within_file_rt_tol)
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
            pc_bin = self._make_precursor_bin(n, prior_masses[n], current_rt, current_intensity, self.within_file_mass_tol, self.within_file_rt_tol)
            bins.append(pc_bin)
            
            prior_mass = (current_mass - self.adduct_sub)/self.adduct_mul + self.adduct_del
            rt_ok = utils.rt_match(current_rt, prior_rts, self.within_file_rt_tol)
            intensity_ok = (current_intensity <= prior_intensities)
            for t in np.arange(len(self.transformations)):
                mass_ok = utils.mass_match(prior_mass[t], prior_masses, self.within_file_mass_tol)
                check = rt_ok*mass_ok*intensity_ok
                # check = rt_ok*mass_ok                
                pos = np.flatnonzero(check)
                possible[n, pos] = t+1
                transformed[n, pos] = prior_mass[t]
                matRT[n, pos] = current_rt            

        print "Total bins=" + str(K) + " total features=" + str(N)
        binning = DiscreteInfo(possible, transformed, matRT, bins, prior_masses, prior_rts)
        return binning         
        
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
        
    def load_model_input(self, input_file, synthetic=False, limit_n=-1, verbose=False):
        """ Load everything that a clustering model requires """

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
                features, corr_mat = self.load_features(file_path, file_id, synthetic=synthetic)
                file_id += 1
                if limit_n > -1:
                    print "Using only " + str(limit_n) + " features from " + file_path
                    features = features[0:limit_n]
                data = PeakData(features, file_path, corr_mat=corr_mat)
                all_features.extend(features)
                data_list.append(data)
                sys.stdout.flush()
            os.chdir(starting_dir)                                
            return data_list
                    
        else:   
                     
            # process only a single file
            features, corr_mat = self.load_features(input_file, synthetic=synthetic)
            if limit_n > -1:
                features = features[0:limit_n]
            data = PeakData(features, input_file, corr_mat=corr_mat)
            return data
                
    def load_features(self, input_file, file_id, load_correlations=False, synthetic=False):

        # first load the features
        features = []
        if input_file.endswith(".csv"):
            features = self.load_features_csv(input_file, file_id)
        elif input_file.endswith(".txt"):
            if synthetic:
                # in SIMA (.txt) format, used for some old synthetic data
                features = self.load_features_sima(input_file, file_id)
            else:
                # in tab-separated format from mzMatch
                features = self.load_features_txt(input_file, file_id)   
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
    
    def load_features_csv(self, input_file, file_id):
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
                    feature = Feature(feature_id, mz, rt, intensity, file_id)
                elif len(elements)==4:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])
                    intensity = utils.num(elements[3])
                    feature = Feature(feature_id, mz, rt, intensity, file_id)
                features.append(feature)
        return features

    def load_features_txt(self, input_file, file_id):
        features = []
        if not os.path.exists(input_file):
            return features
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                feature = Feature(feature_id=feature_id, mass=utils.num(elements[0]), \
                                  rt=utils.num(elements[1]), intensity=utils.num(elements[2]), file_id=file_id)
                features.append(feature)
                feature_id = feature_id + 1
        return features
        
    def load_features_sima(self, input_file, file_id):
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
                feature = Feature(feature_id, mass, rt, intensity, file_id)
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
    
    
def main(argv):    

    loader = FileLoader()
    input_file = '../alignment/input/std1_csv'
    database_file = './database/std1_mols.csv'
    transformation_file = './mulsubs/mulsub2.txt'
    mass_tol = 2
    rt_tol = 2
    across_file_mass_tol = 4

    import time
    start = time.time()
    data_list = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol, across_file_mass_tol)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("TIME TAKEN {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))    
    
if __name__ == "__main__":
   main(sys.argv[1:])

