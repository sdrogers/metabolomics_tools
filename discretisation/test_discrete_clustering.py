from continuous_mass_clusterer import ContinuousGibbs, ContinuousVB
from discrete_mass_clusterer import DiscreteGibbs, DiscreteVB
from models import HyperPars
from preprocessing import FileLoader
import numpy as np
from plotting import ClusterPlotter

def print_stats(peak_data, discrete, continuous):

    discrete_Z = discrete.Z
    cont_Z = continuous.Z
    diff = discrete_Z - cont_Z
    change = np.square(diff.todense()).sum()
    print "Discrete - continuous Z change: " + str(change)
    print 
    
    print 'Discrete'
    cp = ClusterPlotter(peak_data, discrete)
    cp.summary()
    cp.plot_biggest(3)
    print 
    
    print 'Continuous'
    cp = ClusterPlotter(peak_data, continuous)
    cp.summary()
    cp.plot_biggest(3)

def main():

    basedir = '.'
    hp = HyperPars()

    # load synthetic data
#     input_file = basedir + '/input/synthetic/synthdata_0.txt'
#     # input_file = basedir + '/input/synthetic_subset'
#     database_file = basedir + '/database/std1_20_mols.csv'
#     transformation_file = basedir + '/mulsubs/mulsub_synth2.txt'
#     mass_tol = 2
#     rt_tol = 5
#     loader = FileLoader()
#     peak_data = loader.load_model_input(input_file, database_file, transformation_file,
#                                          mass_tol, rt_tol, synthetic=True)

    # load std1 file
    input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
    # input_file = basedir + '/input/std1_csv_subset'    
    database_file = basedir + '/database/std1_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub2.txt'
    mass_tol = 2
    rt_tol = 5
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
           
    # try gibbs sampling
    n_samples = 20
    n_burn = 10
    discrete = DiscreteGibbs(peak_data, hp)
    continuous = ContinuousGibbs(peak_data, hp)
    discrete.n_samples = n_samples
    discrete.n_burn = n_burn
    continuous.n_samples = n_samples
    continuous.n_burn = n_burn
    discrete.run() 
    continuous.run()
    print_stats(peak_data, discrete, continuous)

    # try VB
    n_iters = 20
    discrete = DiscreteVB(peak_data, hp)
    continuous = ContinuousVB(peak_data, hp)
    discrete.n_iterations = n_iters
    continuous.n_iterations = n_iters
    print discrete
    discrete.run()        
    print continuous
    continuous.run()    
    print_stats(peak_data, discrete, continuous)

if __name__ == "__main__": main()