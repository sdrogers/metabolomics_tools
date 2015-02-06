from models import FileLoader
from AdductClusterer import MassBinClusterer

def main():

    # load the std file and transformation_file
    loader = FileLoader()
    basedir = '.'
    input_file = basedir + '/input/std1_csv/std1-file1.identified.csv'    
    transformation_file = basedir + '/mulsub.txt'
    features = loader.load_features(input_file)
    transformations = loader.load_transformation(transformation_file)
    
    # run it through the model
    mbc = MassBinClusterer(features, transformations, 3, 0.1, 15, 100, 0)
    mbc.run()
    
    # plot the result

if __name__ == "__main__": main()