from continuous_mass_clusterer import ContinuousGibbs, ContinuousVB
from discrete_mass_clusterer import DiscreteGibbs, DiscreteVB
from models import FileLoader, HyperPars
import numpy as np
from plotting import ClusterPlotter


def main():

    basedir = '.'
    hp = HyperPars()

    # load synthetic data
    input_file = basedir + '/input/synthetic_small'
    database_file = basedir + '/database/std1_20_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub_synth.txt'
    mass_tol = 2
    rt_tol = 5
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)

#     # load std1 file
#     input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
#     database_file = basedir + '/database/std1_mols.csv'
#     transformation_file = basedir + '/mulsubs/mulsub2.txt'
#     mass_tol = 2
#     rt_tol = 5
#     loader = FileLoader()
#     peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
           
    # try gibbs sampling
    discrete = DiscreteGibbs(peak_data, hp)
    continuous = ContinuousGibbs(peak_data, hp)
    discrete.nsamps = 20
    continuous.nsamps = 20
    discrete.run() 
    continuous.run()

    # try continuous
    discrete = DiscreteVB(peak_data, hp)
    continuous = ContinuousVB(peak_data, hp)
    discrete.n_iterations = 20
    continuous.n_iterations = 20
    print discrete
    discrete.run()        
    print continuous
    continuous.run()  
  
    discrete_Z = discrete.Z
    cont_Z = discrete.Z
    change = (np.square(discrete_Z-cont_Z)).sum()
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

if __name__ == "__main__": main()