# if the discreisation module below is not found, remember to set PYTHONPATH 
# e.g. 
# export PYTHONPATH="/home/joewandy/git/metabolomics_tools:$PYTHONPATH"

from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.models import HyperPars
from discretisation.preprocessing import FileLoader
from discretisation import utils
import numpy as np
import pylab as plt


def load_and_cluster(input_file, database_file, transformation_file, mass_tol, rt_tol):
        print "Loading " + input_file
        loader = FileLoader()        
        peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
        print "Clustering " + input_file
        hp = HyperPars()
        cluster_model = ContinuousVB(peak_data, hp)
        cluster_model.n_iterations = 10
        print cluster_model
        cluster_model.run()    
        return peak_data, cluster_model
    
def get_feature(peak_data, cluster_model, db_entry, t_find):

    cluster_membership = (cluster_model.Z>0.5)
    trans_mat = peak_data.possible.multiply(cluster_membership)
    trans_mat = (trans_mat==t_find+1)    
    trans_count = (trans_mat).sum(0)

    prior_masses = peak_data.prior_masses
    mass_ok = utils.mass_match(db_entry.mass, prior_masses, 10)
    ks = np.flatnonzero(mass_ok)
    
    features = []
    for k in ks:
        if trans_count[k]==0:
            continue
        else:
            peak_idx = np.nonzero(trans_mat[k, :])
            peak_idx = peak_idx.tolist()
            for n in peak_idx:
                f = peak_data.features[n]
                features.append(f)
                
    # if there are multiple features, pick one
    if len(features)==1:
        return features[0]
    elif len(features)>1:
        return features[0]
    else:
        return None

def main():

    basedir = '.'

    dilutions = ["1:1", "1:5", "1:10", "1:50", "1:100", "1:1000"]
    input_files = [
        basedir + '/dilutions_data/Positive/std1/Std1_1.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in5.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in10.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in50.txt',    
        basedir + '/dilutions_data/Positive/std1/Std1_1in100.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in1000.txt'
    ]
    dilutions = ["1:1000"]
    input_files = [
        basedir + '/dilutions_data/Positive/std1/Std1_1in1000.txt'
    ]
    assert len(input_files) == len(dilutions)
    L = len(dilutions)

    database_file = basedir + '/database/std1_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub2.txt'    
    mass_tol = 2
    rt_tol = 5    
    filemap = {}
    for j in range(L):
        dil = dilutions[j]
        input_file = input_files[j]
        load_res = load_and_cluster(input_file, database_file, transformation_file, mass_tol, rt_tol)
        filemap[dil] = load_res

    first_res = filemap["1:1000"]
    first_peak_data = first_res[0]
    database = first_peak_data.database
    transformations = first_peak_data.transformations
    t_find = 6 # M+H
    print "Finding " + str(transformations[t_find])

    D = len(database)
    results = np.zeros((D, L))
    for d in range(D):
        db_entry = database[d]
        for l in range(L):
            dil = dilutions[l]
            load_res = filemap[dil]
            peak_data = load_res[0]
            cluster_model = load_res[1]
            f = get_feature(peak_data, cluster_model, db_entry, t_find)
            print f

if __name__ == "__main__": main()
