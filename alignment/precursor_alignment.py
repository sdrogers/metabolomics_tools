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
    parser.add_argument('-within_file_mass_tol', help='mass tolerance in ppm when binning within the same file', type=float)
    parser.set_defaults(within_file_mass_tol=2.0)
    parser.add_argument('-within_file_rt_tol', help='rt tolerance in seconds when binning within the same file', type=float)
    parser.set_defaults(within_file_rt_tol=5.0)
    parser.add_argument('-within_file_rt_sd', help='standard deviation when clustering by precursor masses within the same file', type=float)
    parser.set_defaults(within_file_rt_sd=2.5)
    parser.add_argument('-across_file_mass_tol', help='mass tolerance in ppm when binning across files', type=float)
    parser.set_defaults(across_file_mass_tol=4.0)
    parser.add_argument('-across_file_rt_tol', help='rt tolerance in seconds when matching peak features across bins in the same cluster but coming from different files', type=float)
    parser.set_defaults(across_file_rt_tol=4.0)
    parser.add_argument('-across_file_rt_sd', help='standard deviation of mixture component when clustering bins by posterior RT across files', type=float)
    parser.set_defaults(across_file_rt_sd=30.0)
    parser.add_argument('-alpha_mass', help='Dirichlet parameter for precursor mass clustering', type=float)
    parser.set_defaults(alpha_mass=100.0)
    parser.add_argument('-alpha_rt', help='Dirichlet Process concentration parameter for mixture on RT', type=float)
    parser.set_defaults(alpha_rt=1.0)
    parser.add_argument('-t', help='threshold for cluster membership for precursor mass clustering', type=float)
    parser.set_defaults(t=0.0)
    parser.add_argument('-mass_clustering_n_iterations', help='no. of iterations for VB precursor clustering', type=int)
    parser.set_defaults(mass_clustering_n_iterations=100)
    parser.add_argument('-rt_clustering_nsamps', help='no. of total samples for Gibbs RT clustering', type=int)
    parser.set_defaults(rt_clustering_nsamps=200)
    parser.add_argument('-rt_clustering_burnin', help='no. of burn-in samples for Gibbs RT clustering', type=int)
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

    input_dir = options.input_dir
    database_file = options.database_file
    transformation_file = options.transformation_file
    gt_file = options.gt_file
    output_path = options.output_path

    alignment_hp.within_file_mass_tol = options.within_file_mass_tol
    alignment_hp.within_file_rt_tol = options.within_file_rt_tol
    alignment_hp.within_file_rt_sd = options.within_file_rt_sd
    alignment_hp.within_file_mass_sd = 1.0/10 # for continuousVB mass clustering
    
    alignment_hp.across_file_mass_tol = options.across_file_mass_tol
    alignment_hp.across_file_rt_tol = options.across_file_rt_tol
    alignment_hp.across_file_rt_sd = options.across_file_rt_sd

    alignment_hp.alpha_mass = options.alpha_mass
    alignment_hp.dp_alpha = options.alpha_rt
    alignment_hp.t = options.t
    
    alignment_hp.mass_clustering_n_iterations = options.mass_clustering_n_iterations
    alignment_hp.rt_clustering_nsamps = options.rt_clustering_nsamps
    alignment_hp.rt_clustering_burnin = options.rt_clustering_burnin
        
    sb = SharedBinMatching(input_dir, database_file, transformation_file, 
                           alignment_hp, synthetic=True, gt_file=gt_file, 
                           verbose=options.verbose, seed=options.seed)
    sb.run(alignment_hp.across_file_mass_tol, alignment_hp.across_file_rt_tol, 
           full_matching=False, show_singleton=True)
    sb.save_output(output_path)
    
if __name__ == "__main__":
   main(sys.argv[1:])
