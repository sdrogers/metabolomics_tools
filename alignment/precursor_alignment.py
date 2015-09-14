#!/usr/bin/python

import sys
import argparse

from shared_bin_matching import SharedBinMatching
from models import HyperPars as AlignmentHyperPars

def print_banner():
    '''Prints some banner at program start'''
    print "---------------------------------------------------------------"
    print "precursor_alignment.py -- precursor clustering and alignment"
    print "---------------------------------------------------------------"
    print
    
def get_options(argv):
    '''Parses command-line options'''
    parser = argparse.ArgumentParser()

    # core arguments
    parser.add_argument('-i', required=True, dest='input_dir', help='input directory')
    parser.add_argument('-o', required=True, dest='output_path', help='output path')
    parser.add_argument('-trans', required=True, dest="transformation_file", help='transformation file')
    parser.add_argument('-db', dest='database_file', help='identification database file')
    parser.set_defaults(database_file=None)
    parser.add_argument('-gt', dest='gt_file', help='ground truth file')
    parser.set_defaults(gt_file=None)
    parser.add_argument('-v', dest='verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)
    parser.add_argument('-seed', help='random seed. Set this to get the same results each time.', type=float)
    parser.set_defaults(seed=-1) # default is not seeded

    # precursor clustering and alignment arguments
    parser.add_argument('-within_file_binning_mass_tol', help='mass tolerance for binning.', type=float)
    parser.set_defaults(binning_mass_tol=2.0)
    parser.add_argument('-within_file_binning_rt_tol', help='RT tolerance for binning.', type=float)
    parser.set_defaults(binning_rt_tol=5.0)
    parser.add_argument('-within_file_rt_sd', help='Within-file RT standard deviation.', type=float)
    parser.set_defaults(within_file_rt_sd=2.5)
    parser.add_argument('-across_file_binning_mass_tol', help='mass tolerance for binning.', type=float)
    parser.set_defaults(across_file_binning_mass_tol=4.0)
    parser.add_argument('-across_file_rt_sd', help='Across-file RT standard deviation.', type=float)
    parser.set_defaults(across_file_rt_sd=30.0)
    parser.add_argument('-alpha_mass', help='Dirichlet prior for precursor mass clustering in the same file.', type=float)
    parser.set_defaults(alpha_mass=100.0)
    parser.add_argument('-alpha_rt', help='Dirichlet process prior for RT clustering across files.', type=float)
    parser.set_defaults(alpha_rt=1.0)
    parser.add_argument('-t', help='Threshold for precursor cluster membership.', type=float)
    parser.set_defaults(t=0.25)
    parser.add_argument('-mass_clustering_n_iterations', help='No. of iterations for variational inference for precursor clustering in the same file.', type=int)
    parser.set_defaults(mass_clustering_n_iterations=100)
    parser.add_argument('-rt_clustering_nsamps', help='Total no. of samples for RT clustering across files.', type=int)
    parser.set_defaults(rt_clustering_nsamps=200)
    parser.add_argument('-rt_clustering_burnin', help='No. of burn-in samples for RT clustering across files.', type=int)
    parser.set_defaults(rt_clustering_burnin=100)
    
    # parse it
    options = parser.parse_args(argv)
    if options.verbose:
        print "Options", options
    return options
    
def main(argv):    

    print_banner()
    options = get_options(argv)
    alignment_hp = AlignmentHyperPars()    

#     input_dir = './input/P1/100'
#     database_file = '../discretisation/database/std1_mols.csv'
#     transformation_file = '../discretisation/mulsubs/mulsub2.txt'
#     gt_file = './input/P1/ground_truth/ground_truth_100.txt'
#     
#     alignment_hp.binning_mass_tol = 2
#     alignment_hp.binning_rt_tol = 5
#     alignment_hp.within_file_rt_sd = 2.5
#     alignment_hp.across_file_rt_sd = 30
#     alignment_hp.alpha_mass = 100.0
#     alignment_hp.alpha_rt = 100.0
#     alignment_hp.t = 0.25
#     alignment_hp.mass_clustering_n_iterations = 100
#     alignment_hp.rt_clustering_nsamps = 200
#     alignment_hp.rt_clustering_burnin = 100

    input_dir = options.input_dir
    database_file = options.database_file
    transformation_file = options.transformation_file
    gt_file = options.gt_file
    output_path = options.output_path

    alignment_hp.binning_mass_tol = options.within_file_binning_mass_tol
    alignment_hp.binning_rt_tol = options.within_file_binning_rt_tol
    alignment_hp.across_file_mass_tol = options.across_file_binning_mass_tol
    alignment_hp.within_file_rt_sd = options.within_file_rt_sd
    alignment_hp.across_file_rt_sd = options.across_file_rt_sd
    alignment_hp.alpha_mass = options.alpha_mass
    alignment_hp.alpha_rt = options.alpha_rt
    alignment_hp.t = options.t
    alignment_hp.mass_clustering_n_iterations = options.mass_clustering_n_iterations
    alignment_hp.rt_clustering_nsamps = options.rt_clustering_nsamps
    alignment_hp.rt_clustering_burnin = options.rt_clustering_burnin
    alignment_hp.mass_sd = 1.0/10 # for continuousVB mass clustering
        
    sb = SharedBinMatching(input_dir, database_file, transformation_file, 
                           alignment_hp, synthetic=True, gt_file=gt_file, verbose=options.verbose, seed=options.seed)
    sb.run(show_singleton=True)
    sb.save_output(output_path)
    
if __name__ == "__main__":
   main(sys.argv[1:])
