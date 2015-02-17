from models import FileLoader, HyperPars
from discrete_mass_clusterer import DiscreteGibbs

def main():

    basedir = '.'

    # load synthetic data
    input_file = basedir + '/input/synthetic/synthdata_0.txt'
    database_file = basedir + '/database/std1_20_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub_synth.txt'
    mass_tol = 2
    rt_tol = 5
    loader = FileLoader()
    peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
           
    # run it through the model
    hp = HyperPars()
    nsamps = 20
    mbc = DiscreteGibbs(peak_data, hp, nsamps)
    mbc.run()

#     # load std1 file
#     input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
#     database_file = basedir + '/database/std1_mols.csv'
#     transformation_file = basedir + '/mulsubs/mulsub.txt'
#     mass_tol = 2
#     rt_tol = 5
#     loader = FileLoader()
#     peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
#        
#     # run it through the model
#     hp = HyperPars()
#     nsamps = 20
#     mbc = DiscreteGibbs(peak_data, hp, nsamps)
#     mbc.run()
    
if __name__ == "__main__": main()