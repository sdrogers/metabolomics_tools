from continuous_mass_clusterer import ContinuousGibbs, ContinuousVB
from discrete_mass_clusterer import DiscreteGibbs, DiscreteVB
from models import FileLoader, HyperPars
from plotting import ClusterPlotter
import numpy as np
import pylab as plt
import scipy.sparse as sp


def main():

    basedir = '.'
    hp = HyperPars()

#     # load synthetic data
#     input_file = basedir + '/input/synthetic/synthdata_0.txt'
#     database_file = basedir + '/database/std1_20_mols.csv'
#     transformation_file = basedir + '/mulsubs/mulsub_synth.txt'
#     mass_tol = 2
#     rt_tol = 5
#     loader = FileLoader()
#     peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)

    # load std1 file
    input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
    database_file = basedir + '/database/std1_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub.txt'
    mass_tol = 2
    rt_tol = 5
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
           
#     # try gibbs sampling
#     mbc = DiscreteGibbs(peak_data, hp)
#     mbc.nsamps = 20
#     mbc.run()
# 
#     vb = ContinuousGibbs(peak_data, hp)
#     vb.nsamps = 20
#     vb.run()

    # try vb
    mbc = DiscreteVB(peak_data, hp)
    vb = ContinuousVB(peak_data, hp)
    mbc.n_iterations = 20
    vb.n_iterations = 20
    print mbc
    mbc.run()        
    print vb
    vb.run()  
 
    # plot vb results
    I = np.identity(peak_data.num_peaks)
    discrete_Z = mbc.Z.todense() - I
    cont_Z = vb.Z.todense() - I
    discrete_Z = sp.csr_matrix(discrete_Z)
    cont_Z = sp.csr_matrix(cont_Z)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.spy(discrete_Z)
    plt.title('discrete QZ')
    plt.subplot(1, 2, 2)
    plt.spy(cont_Z)
    plt.title('continous QZ')
    plt.show()    
    Q_change = (np.square(discrete_Z-cont_Z)).sum()
    print "Discrete - continuous QZ change: " + str(Q_change)
    
    print 'Discrete'
    cp = ClusterPlotter(peak_data, mbc)
    cp.summary()
    cp.plot_biggest(3)

    print
    print 'Continuous'
    cp = ClusterPlotter(peak_data, vb)
    cp.summary()
    cp.plot_biggest(3)

if __name__ == "__main__": main()