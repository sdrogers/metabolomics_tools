from discretisation import utils
from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.models import HyperPars
from discretisation.preprocessing import FileLoader
import numpy as np


class DilutionExporter(object):
    
    def __init__(self, stdfile, mass_tol=2, rt_tol=5, identify_tol=2):

        self.dilutions = ["1:1", "1:5", "1:10", "1:50", "1:100", "1:1000"]
        self.L = len(self.dilutions)  
        self.stdfile = stdfile.lower()
        self.input_files = [
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1.csv',
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1in5.csv',
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1in10.csv',
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1in50.csv',
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1in100.csv',
            './dilutions_data/Positive/' + self.stdfile + '/' + self.stdfile.capitalize() + '_1in1000.csv'
        ]
        assert len(self.input_files) == len(self.dilutions)
        self.database_file = './database/' + self.stdfile + '_mols.csv'
        self.transformation_file = './mulsubs/mulsub2.txt'    
        
        # for binning
        self.mass_tol = mass_tol
        self.rt_tol = rt_tol
        
        # for identification
        self.identify_tol = identify_tol
        
        # load the files
        L = len(self.dilutions)    
        self.filemap = {}
        for j in range(L):
            dil = self.dilutions[j]
            input_file = self.input_files[j]
            load_res = self._load_and_cluster(input_file)
            self.filemap[dil] = load_res    
            
        # get the database and transformations objects from any loaed file
        any_entry = self.filemap[self.filemap.keys()[0]]
        any_peak_data = any_entry[0]
        self.database = any_peak_data.database
        self.transformations = any_peak_data.transformations   
        self.D = len(self.database)
        self.T = len(self.transformations)                
        
    def save_files(self):      
        
        for t in range(len(self.transformations)):
            # results is an np array of features
            results = self._get_results(t)
            # extract the intensity, m/z and rt values from results
            intenses = self._to_attribute_array(results, 'intensity')
            masses = self._to_attribute_array(results, 'mass')
            rts = self._to_attribute_array(results, 'rt')
            # save them all
            fname = './transformations_data/' + self.stdfile + '/' + str(t) + '.txt'
            print "Saving to " + fname
            np.savetxt(fname, intenses)
            fname = './transformations_data/' + self.stdfile + '/' + str(t) + '.mass.txt'
            print "Saving to " + fname
            np.savetxt(fname, masses)
            fname = './transformations_data/' + self.stdfile + '/' + str(t) + '.rt.txt'
            print "Saving to " + fname
            np.savetxt(fname, rts)
            
    def _load_and_cluster(self, input_file):
            print "Loading " + input_file
            loader = FileLoader()        
            peak_data = loader.load_model_input(input_file, self.database_file, self.transformation_file, 
                                                self.mass_tol, self.rt_tol, make_bins=True)
            print "Clustering " + input_file
            hp = HyperPars()
            cluster_model = ContinuousVB(peak_data, hp)
            cluster_model.n_iterations = 20
            print cluster_model
            cluster_model.run()    
            return peak_data, cluster_model

    def _get_feature(self, peak_data, cluster_model, db_entry, t_find, identify_tol):
    
        # find cluster membership for all peaks with transformation t_find
        cluster_membership = (cluster_model.Z>0.5)
        trans_mat = peak_data.possible.multiply(cluster_membership)
        trans_mat = (trans_mat==t_find+1)    
        trans_count = (trans_mat).sum(0)
    
        # match prior bin masses against database entry's mass
        prior_masses = peak_data.prior_masses
        mass_ok = utils.mass_match(db_entry.mass, prior_masses, identify_tol)
        ks = np.flatnonzero(mass_ok)
        
        # enumerate all the possible features
        features = []
        for k in ks:
            if trans_count[0, k]==0:
                continue
            else:
                temp = np.nonzero(trans_mat[:, k])
                peak_idx = temp[0]
                for n in peak_idx:
                    f = peak_data.features[n]
                    features.append(f)
                    
        # if there are multiple features, pick the most intense one
        if len(features)==1:
            return features[0]
        elif len(features)>1: 
            intenses = np.array([f.intensity for f in features])
            most_intense = intenses.argmax()
            return features[most_intense]
        else:
            return None
        
    def _get_results(self, t_find):
            
        print "Finding " + str(self.transformations[t_find])
        results = np.zeros((self.D, self.L), dtype=object)   
    
        # for all db entries
        for d in range(self.D):
            db_entry = self.database[d]
            # for all dilution files
            for l in range(self.L):
                dil = self.dilutions[l]
                # find the peak
                load_res = self.filemap[dil]
                peak_data = load_res[0]
                cluster_model = load_res[1]
                f = self._get_feature(peak_data, cluster_model, db_entry, 
                                     t_find, self.identify_tol)
                results[d, l] = f
        return results       
    
    def _to_attribute_array(self, results, attribute):
        value_arr = np.zeros_like(results)
        for (i,j), feature in np.ndenumerate(results):
            if feature is None:
                value = 0
            else:
                value = getattr(feature, attribute)
            value_arr[i, j] = value
        return value_arr 