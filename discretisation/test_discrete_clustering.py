from continuous_mass_clusterer import ContinuousVB
from discrete_mass_clusterer import DiscreteGibbs, DiscreteVB
from models import FileLoader, HyperPars
import numpy as np
import pylab as plt
import scipy.sparse as sp


def main():

    basedir = '.'

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
           
    # try gibbs sampling
    hp = HyperPars()
    mbc = DiscreteGibbs(peak_data, hp)
    mbc.nsamps = 20
    mbc.run()

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
    mbc_QZ = mbc.QZ.todense() - I
    vb_QZ = vb.QZ.todense() - I
    mbc_QZ = sp.csr_matrix(mbc_QZ)
    vb_QZ = sp.csr_matrix(vb_QZ)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.spy(mbc_QZ)
    plt.title('discrete QZ')
    plt.subplot(1, 2, 2)
    plt.spy(vb_QZ)
    plt.title('continous QZ')
    plt.show()    
    Q_change = (np.square(mbc_QZ-vb_QZ)).sum()
    print "Discrete - continuous QZ change: " + str(Q_change)

if __name__ == "__main__": main()