import os
import sys
basedir = '..'
sys.path.append(basedir)

import numpy as np
import pandas as pd

import cPickle
import random
import copy
import glob
import gzip

from alignment.models import HyperPars as AlignmentHyperPars
from discretisation.adduct_cluster import AdductCluster, Peak, Possible
from discretisation import utils
from discretisation.preprocessing import FileLoader
from alignment.shared_bin_matching import SharedBinMatching as Aligner
from alignment.ground_truth import GroundTruth

def load_or_create_clustering(filename, input_dir, transformation_file, hp):
    try:
        with gzip.GzipFile(filename, 'rb') as f:
            combined_list = cPickle.load(f)
            print "Loaded from %s" % filename
            return combined_list
    except (IOError, EOFError):
        loader = FileLoader()        
        data_list = loader.load_model_input(input_dir, synthetic=True)
        aligner = Aligner(data_list, None, transformation_file, 
                               hp, verbose=False, seed=1234567890, parallel=False, mh_biggest=True, use_vb=False)
        clustering_results = aligner._first_stage_clustering()
        combined_list = zip(data_list, clustering_results)
        with gzip.GzipFile(filename, 'wb') as f:
            cPickle.dump(combined_list, f, protocol=cPickle.HIGHEST_PROTOCOL)        
        print "Saved to %s" % filename
        return combined_list
        
def train(selected_data, param_list, hp, match_mode, evaluation_method, transformation_file, gt_file):
    
    performances = []
    for param in param_list:

        if len(param) > 2:
            hp.across_file_mass_tol = param[0]
            hp.across_file_rt_tol = param[1]
            hp.within_file_rt_tol = param[2]
            hp.matching_alpha = param[3]
        else:
            hp.across_file_mass_tol = param[0]
            hp.across_file_rt_tol = param[1]
            
        selected_files = [x[0] for x in selected_data]  
        selected_clusterings = [x[1] for x in selected_data]            
        aligner = Aligner(selected_files, None, transformation_file, 
                               hp, verbose=False, seed=1234567890)
        aligner.run(match_mode, first_stage_clustering_results=selected_clusterings)

        res = aligner.evaluate_performance(gt_file, verbose=False, print_TP=True, method=evaluation_method)
        output = param+res[0]
        if len(param) > 2:
            print "mass_tol=%d, rt_tol=%d, grouping_tol=%d, matching_alpha=%.1f, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output        
        else:
            print "mass_tol=%d, rt_tol=%d, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output                    
        performances.append(output)

    if len(param) > 2:
        df = pd.DataFrame(performances, columns=['mass_tol', 'rt_tol', 'grouping_tol', 'matching_alpha', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold'])
        sorted_df = df.sort_values(['F1', 'mass_tol', 'rt_tol', 'grouping_tol', 'matching_alpha'], ascending=[False, True, True, True, True])
    else:
        df = pd.DataFrame(performances, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold'])    
        sorted_df = df.sort_values(['F1', 'mass_tol', 'rt_tol'], ascending=[False, True, True])
        
    best_row = sorted_df.iloc[0]
    return df, best_row
    
def test(selected_data, best_row, hp, match_mode, evaluation_method, transformation_file, gt_file):

    if 'grouping_tol' in best_row:
        param = (best_row['mass_tol'], best_row['rt_tol'], best_row['grouping_tol'], best_row['matching_alpha'])
        hp.across_file_mass_tol = param[0]
        hp.across_file_rt_tol = param[1]
        hp.within_file_mass_tol = param[2]
        hp.matching_alpha = param[3]
    else:
        param = (best_row['mass_tol'], best_row['rt_tol'])
        hp.across_file_mass_tol = param[0]
        hp.across_file_rt_tol = param[1]
        
    selected_files = [x[0] for x in selected_data]
    selected_clusterings = [x[1] for x in selected_data]    
    aligner = Aligner(selected_files, None, transformation_file, 
                           hp, verbose=False, seed=1234567890)
    aligner.run(match_mode, first_stage_clustering_results=selected_clusterings)

    res = aligner.evaluate_performance(gt_file, verbose=False, print_TP=True, method=evaluation_method)
    output = param+res[0]
    return output
    
def train_test_single(match_mode, training_data, testing_data, i, param_list, hp, evaluation_method, transformation_file, gt_file):
    
    print "Iteration %d" % i
    print "Training on %s" % [x[0].filename for x in training_data]
    training_df, best_training_row = train(training_data, param_list, hp, match_mode, evaluation_method, transformation_file, gt_file)
    print "Best row is\n%s" % best_training_row

    print "Testing on %s" % [x[0].filename for x in testing_data]
    match_res = test(testing_data, best_training_row, hp, match_mode, evaluation_method, transformation_file, gt_file)
    output = (match_mode,) + match_res
    if len(output) == 10:
        print "match_mode=%d, mass_tol=%d, rt_tol=%d, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output
    else:
        print "match_mode=%d, mass_tol=%d, rt_tol=%d, grouping_tol=%d, matching_alpha=%.3f, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output
        
    item = (training_data, training_df, best_training_row, match_res)
    return item
    
def train_test(match_mode, training_list, testing_list, param_list, hp, evaluation_method, transformation_file, gt_file):
    assert len(training_list) == len(testing_list)
    n_iter = len(training_list)
    exp_results = []
    for i in range(n_iter):
        training_data = training_list[i]
        testing_data = testing_list[i]
        item = train_test_single(match_mode, training_data, testing_data, i, param_list, hp, evaluation_method, transformation_file, gt_file)
        exp_results.append(item)
        print
        
    return exp_results
    
def run_experiment_single(match_mode, training_list, testing_list, i, param_list, filename, hp, evaluation_method, transformation_file, gt_file):
    try:
        with gzip.GzipFile(filename, 'rb') as f:        
            item = cPickle.load(f)
            print "Loaded from %s" % filename
            return item
    except (IOError, EOFError):
        training_data = training_list[i]
        testing_data = testing_list[i]
        item = train_test_single(match_mode, training_data, testing_data, i, param_list, hp, evaluation_method, transformation_file, gt_file)        
        with gzip.GzipFile(filename, 'wb') as f:
            cPickle.dump(item, f, protocol=cPickle.HIGHEST_PROTOCOL)                        
        print "Saved to %s" % filename
    return item
    
def run_experiment(match_mode, training_list, testing_list, param_list, filename, hp, evaluation_method, transformation_file, gt_file):
    try:
        with gzip.GzipFile(filename, 'rb') as f:        
            exp_results = cPickle.load(f)
            print "Loaded from %s" % filename
            return exp_results
    except (IOError, EOFError):
        exp_results = train_test(match_mode, training_list, testing_list, param_list, hp, evaluation_method, transformation_file, gt_file)
        with gzip.GzipFile(filename, 'wb') as f:
            cPickle.dump(exp_results, f, protocol=cPickle.HIGHEST_PROTOCOL)                        
        print "Saved to %s" % filename
    return exp_results
    
def load_or_create_filelist(filename, combined_list, n_iter, n_files):
    try:
        with gzip.GzipFile(filename, 'rb') as f:        
            item_list = cPickle.load(f)
            print "Loaded from %s" % filename
            for item in item_list:
                print "%s" % [x[0].filename for x in item]
            return item_list
    except (IOError, EOFError):
        item_list = []
        for i in range(n_iter):
            item = random.sample(combined_list, n_files)
            print "%s" % [x[0].filename for x in item]
            item_list.append(item)
        with gzip.GzipFile(filename, 'wb') as f:
            cPickle.dump(item_list, f, protocol=cPickle.HIGHEST_PROTOCOL)                    
        print "Saved to %s" % filename
        return item_list
