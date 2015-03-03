# if the discreisation module below is not found, remember to set PYTHONPATH 
# e.g. 
# export PYTHONPATH="/home/joewandy/git/metabolomics_tools:$PYTHONPATH"

from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.models import HyperPars
from discretisation.plotting import ClusterPlotter
from discretisation.preprocessing import FileLoader


def main():

    basedir = '.'
    database_file = basedir + '/database/std1_mols.csv'
    transformation_file = basedir + '/mulsubs/mulsub2.txt'
    input_files = [
        basedir + '/dilutions_data/Positive/std1/Std1_1.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in5.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in10.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in50.txt',    
        basedir + '/dilutions_data/Positive/std1/Std1_1in100.txt',
        basedir + '/dilutions_data/Positive/std1/Std1_1in1000.txt'
    ]

    loader = FileLoader()        
    mass_tol = 2
    rt_tol = 5

    hp = HyperPars()
    n_iters = 20

    for input_file in input_files:
        print "Loading " + input_file
        peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol)
        print "Clustering " + input_file
        continuous = ContinuousVB(peak_data, hp)
        continuous.n_iterations = n_iters
        print continuous
        continuous.run()    
        cp = ClusterPlotter(peak_data, continuous)
        cp.summary()
        cp.plot_biggest(3)

if __name__ == "__main__": main()
