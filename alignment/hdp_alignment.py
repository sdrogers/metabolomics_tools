from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.identification import MolAnnotator
from discretisation.models import HyperPars
from discretisation.preprocessing import FileLoader
import numpy as np
import pylab as plt


def main():

    database = '../discretisation/database/std1_mols.csv'
    transformation = '../discretisation/mulsubs/mulsub.txt'
    input_file = './input/std1_csv_2'
    mass_tol = 5
    rt_tol = 20
    identify_tol = 10

    # load the std file, database_file molecules and transformation_file
    loader = FileLoader()
    data_list = loader.load_model_input(input_file, database, transformation, mass_tol, rt_tol)
    for peak_data in data_list:

        # cluster each peak data by mass
        hp = HyperPars()
        cluster_model = ContinuousVB(peak_data, hp)
        cluster_model.n_iterations = 20
        print cluster_model
        cluster_model.run()        
        peak_data.cluster_model = cluster_model
        
        # identify each bins in peak data
        ann = MolAnnotator()    
        moldb = peak_data.database
        bins = peak_data.bins
        ann.identify_bins(moldb, bins)
        
    # do HDP on the cluster RT for each peak_data
    
    

if __name__ == "__main__": main()