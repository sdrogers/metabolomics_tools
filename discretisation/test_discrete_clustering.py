from models import FileLoader
from AdductClusterer import MassBinClusterer

def main():

    loader = FileLoader()
    basedir = '.'

    # load synthetic data
    database_file = basedir + '/database/std1_20_mols.csv'
    database = loader.load_database(database_file)
    transformation_file = basedir + '/mulsub_synth.txt'
    transformations = loader.load_transformation(transformation_file)
    input_file = basedir + '/input/synthetic/synthdata_0.txt'
    features = loader.load_features_sima(input_file)
      
    # run it through the model
    mass_tol = 100
    alpha = 0.01
    sigma = 20
    nsamps = 20
    mbc = MassBinClusterer(features, database, transformations, mass_tol, alpha, sigma, nsamps)
    mbc.run()

#     # load std1 file
#     database_file = basedir + '/database/std1_mols.csv'
#     database = loader.load_database(database_file)
#     transformation_file = basedir + '/mulsub.txt'
#     transformations = loader.load_transformation(transformation_file)
#     input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
#     features = loader.load_features(input_file)
#   
#     # run it through the model
#     mass_tol = 100
#     alpha = 0.01
#     sigma = 20
#     nsamps = 20
#     mbc = MassBinClusterer(features, database, transformations, mass_tol, alpha, sigma, nsamps)
#     mbc.run()
    
if __name__ == "__main__": main()