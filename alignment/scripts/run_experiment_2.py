#!/usr/bin/env python

import os
import sys
import argparse

basedir = '../..'
sys.path.append(basedir)

from alignment.models import HyperPars as AlignmentHyperPars
from alignment.experiment import *

def get_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', required=True, dest='iter', help='Iteration', type=int)
    options = parser.parse_args(argv)
    print "Options", options
    return options
    
def main(argv):    

    options = get_options(argv)

    input_dir = '../input/std1_csv_full_old'
    transformation_file = '../pos_transformations_full.yml'
    gt_file = '../input/std1_csv_full_old/ground_truth/ground_truth.txt'

    hp = AlignmentHyperPars()    
    hp.within_file_mass_tol = 5
    hp.within_file_rt_tol = 30
    hp.across_file_mass_tol = 10
    hp.across_file_rt_tol = 60
    hp.alpha_mass = 1.0
    hp.dp_alpha = 1000.0
    hp.beta = 0.1
    hp.t = 0.0
    hp.mass_clustering_n_iterations = 200
    hp.rt_clustering_nsamps = 100
    hp.rt_clustering_burnin = 0

    print hp

    evaluation_method = 2
    n_iter = 30
    
    param_list = []
    for mass_tol in range(2, 11, 2):
        for rt_tol in range(5, 101, 5):
            param_list.append((mass_tol, rt_tol))
            
    param_list_mwg = []
    for mass_tol in range(2, 11, 2):
        for rt_tol in range(5, 101, 5):
            for group_tol in range(2, 11, 2):
                for alpha in range(0, 11, 2):
                    param_list_mwg.append((mass_tol, rt_tol, group_tol, alpha/10.0))
                    
    param_list_mwg = param_list_mwg[0:2]
                    
    n_files = 2
    training_list = load_or_create_filelist('../notebooks/pickles/training_list_2.p', None, n_iter, n_files)    
    testing_list = load_or_create_filelist('../notebooks/pickles/testing_list_2.p', None, n_iter, n_files)    
    outfile = '../notebooks/pickles/res_mwg_2_iter_%d.p' % options.iter
    exp_results = run_experiment_single(3, training_list, testing_list, options.iter, param_list_mwg, outfile, hp, evaluation_method, transformation_file, gt_file)
    
if __name__ == "__main__":
   main(sys.argv[1:])
