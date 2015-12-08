#!/usr/bin/python

import sys
import argparse
import time

from shared_bin_matching import SharedBinMatching
from models import HyperPars as AlignmentHyperPars
from discretisation.preprocessing import FileLoader


def print_banner():
    '''Prints some banner at program start'''
    print "---------------------------------------------------------------"
    print "precursor_alignment.py -- precursor clustering and alignment"
    print "---------------------------------------------------------------"
    print
    
def get_options(argv):
    '''Parses command-line options'''
    parser = argparse.ArgumentParser()

    # default parameters
    alignment_hp = AlignmentHyperPars()    

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
    parser.add_argument('-match_mode', dest='match_mode', help='Matching mode', type=int)
    parser.set_defaults(match_mode=0)
    parser.add_argument('-seed', help='random seed. Set this to get the same results each time.', type=float)
    parser.set_defaults(seed=-1) # default is not seeded

    # precursor clustering and alignment arguments
    parser.add_argument('-within_file_mass_tol', required=True, help='mass tolerance in ppm when binning within the same file', type=float)
    parser.set_defaults(within_file_mass_tol=alignment_hp.within_file_mass_tol)
    parser.add_argument('-within_file_rt_tol', help='rt tolerance in seconds when binning within the same file', type=float)
    parser.set_defaults(within_file_rt_tol=alignment_hp.within_file_rt_tol)
    parser.add_argument('-across_file_mass_tol', help='mass tolerance in ppm when binning across files', type=float)
    parser.set_defaults(across_file_mass_tol=alignment_hp.across_file_mass_tol)
    parser.add_argument('-across_file_rt_tol', help='rt tolerance in seconds when matching peak features across bins in the same cluster but coming from different files', type=float)
    parser.set_defaults(across_file_rt_tol=alignment_hp.across_file_rt_tol)
    parser.add_argument('-alpha_mass', help='Dirichlet parameter for precursor mass clustering', type=float)
    parser.set_defaults(alpha_mass=alignment_hp.alpha_mass)
    parser.add_argument('-alpha_rt', help='Dirichlet Process concentration parameter for mixture on RT', type=float)
    parser.set_defaults(alpha_rt=alignment_hp.dp_alpha)
    parser.add_argument('-t', help='threshold for cluster membership for precursor mass clustering', type=float)
    parser.set_defaults(t=alignment_hp.t)
    parser.add_argument('-mass_clustering_n_iterations', help='no. of iterations for VB precursor clustering', type=int)
    parser.set_defaults(mass_clustering_n_iterations=alignment_hp.mass_clustering_n_iterations)
    parser.add_argument('-rt_clustering_nsamps', help='no. of total samples for Gibbs RT clustering', type=int)
    parser.set_defaults(rt_clustering_nsamps=alignment_hp.rt_clustering_nsamps)
    parser.add_argument('-rt_clustering_burnin', help='no. of burn-in samples for Gibbs RT clustering', type=int)
    parser.set_defaults(rt_clustering_burnin=alignment_hp.rt_clustering_burnin)
    
    # parse it
    options = parser.parse_args(argv)
    if options.verbose:
        print "Options", options

    return options, alignment_hp
    
def main(argv):    

    print_banner()
    options, alignment_hp = get_options(argv)

    input_dir = options.input_dir
    database_file = options.database_file
    transformation_file = options.transformation_file
    gt_file = options.gt_file
    output_path = options.output_path
    match_mode = options.match_mode

    alignment_hp.within_file_mass_tol = options.within_file_mass_tol
    alignment_hp.within_file_rt_tol = options.within_file_rt_tol    
    alignment_hp.across_file_mass_tol = options.across_file_mass_tol
    alignment_hp.across_file_rt_tol = options.across_file_rt_tol
    alignment_hp.alpha_mass = options.alpha_mass
    alignment_hp.dp_alpha = options.alpha_rt
    alignment_hp.t = options.t
    alignment_hp.mass_clustering_n_iterations = options.mass_clustering_n_iterations
    alignment_hp.rt_clustering_nsamps = options.rt_clustering_nsamps
    alignment_hp.rt_clustering_burnin = options.rt_clustering_burnin
        
    loader = FileLoader()        
    data_list = loader.load_model_input(input_dir, synthetic=True, verbose=options.verbose)    
    sb = SharedBinMatching(data_list, database_file, transformation_file, 
                           alignment_hp, verbose=options.verbose, seed=options.seed)
    sb.run(match_mode)
    sb.save_output(output_path)
    sb.save_project(output_path + ".project")
    sys.stdout.flush()
#     print"Ending program in 60 seconds ..."
#     time.sleep(60)
    
if __name__ == "__main__":
   main(sys.argv[1:])
