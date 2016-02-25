import os
import sys
basedir = '..'
sys.path.append(basedir)

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

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
        
def train(selected_data, param_list, hp, match_mode, evaluation_method, transformation_file, gt_file, q=2):
    
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

        res = aligner.evaluate_performance(gt_file, verbose=False, print_TP=True, method=evaluation_method, q=q)
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
    
def test(selected_data, best_row, hp, match_mode, evaluation_method, transformation_file, gt_file, q=2):

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

    res = aligner.evaluate_performance(gt_file, verbose=False, print_TP=True, method=evaluation_method, q=q)
    output = param+res[0]
    return output
    
def train_test_single(match_mode, training_data, testing_data, i, param_list, hp, evaluation_method, transformation_file, gt_file, q=2):
    
    print "Iteration %d" % i
    print "Training on %s" % [x[0].filename for x in training_data]
    training_df, best_training_row = train(training_data, param_list, hp, match_mode, evaluation_method, transformation_file, gt_file, q=q)
    print "Best row is\n%s" % best_training_row

    print "Testing on %s" % [x[0].filename for x in testing_data]
    match_res = test(testing_data, best_training_row, hp, match_mode, evaluation_method, transformation_file, gt_file, q=q)
    output = (match_mode,) + match_res
    if len(output) == 10:
        print "match_mode=%d, mass_tol=%d, rt_tol=%d, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output
    else:
        print "match_mode=%d, mass_tol=%d, rt_tol=%d, grouping_tol=%d, matching_alpha=%.3f, tp=%d, fp=%d, fn=%d, prec=%.3f, rec=%.3f, f1=%.3f, th_prob=%.3f" % output
        
    item = (training_data, training_df, best_training_row, match_res)
    return item
    
def train_test(match_mode, training_list, testing_list, param_list, hp, evaluation_method, transformation_file, gt_file, q=2):
    assert len(training_list) == len(testing_list)
    n_iter = len(training_list)
    exp_results = []
    for i in range(n_iter):
        training_data = training_list[i]
        testing_data = testing_list[i]
        item = train_test_single(match_mode, training_data, testing_data, i, param_list, hp, evaluation_method, transformation_file, gt_file, q=q)
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
    
def run_experiment(match_mode, training_list, testing_list, param_list, filename, hp, evaluation_method, transformation_file, gt_file, q=2):
    try:
        with gzip.GzipFile(filename, 'rb') as f:        
            exp_results = cPickle.load(f)
            print "Loaded from %s" % filename
            return exp_results
    except (IOError, EOFError):
        exp_results = train_test(match_mode, training_list, testing_list, param_list, hp, evaluation_method, transformation_file, gt_file, q=q)
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
        
def load_results(path, n_iter):
    results = []
    for i in range(n_iter):
        filename = path % i
        with gzip.GzipFile(filename, 'rb') as f:        
            item = cPickle.load(f)
            results.append(item)
            print "Loaded from %s" % filename
    return results        
        
def replace_clustering(combined_list, item_list):

    combined_map = {}
    for peakdata, clustering in combined_list:
        combined_map[peakdata.filename] = clustering

    new_item_list = []
    for row in item_list:
        new_row = []
        for peakdata, clustering in row:
            new_clustering = combined_map[peakdata.filename]
            new_item = (peakdata, new_clustering)
            new_row.append(new_item)
        new_item_list.append(new_row)

    return new_item_list
    
def evaluate_performance(hp, aligner, gt_file, evaluation_method, q=2):
    param = (hp.across_file_mass_tol, hp.across_file_rt_tol )
    res = aligner.evaluate_performance(gt_file, verbose=False, print_TP=True, method=evaluation_method, q=q)
    performances = []
    for r in res:
        performances.append(param+r)
    df = pd.DataFrame(performances, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold'])
    return df
    
def second_stage_clustering(hp, training_list, i, evaluation_method, transformation_file, gt_file, clustering_out=None, df_out=None, use_adduct_likelihood=True, parallel=False, q=2):

    hp.second_stage_clustering_use_adduct_likelihood = use_adduct_likelihood    
    print hp
    
    training_data = training_list[i]
    print "Iteration %d" % i
    print "Training on %s" % [x[0].filename for x in training_data]

    selected_files = [x[0] for x in training_data]  
    selected_clusterings = [x[1] for x in training_data]            
    aligner = Aligner(selected_files, None, transformation_file, 
                           hp, verbose=False, seed=1234567890, parallel=parallel)
    match_mode = 2
    aligner.run(match_mode, first_stage_clustering_results=selected_clusterings)
    
    if clustering_out is not None:
        with gzip.GzipFile(clustering_out, 'wb') as f:
            cPickle.dump(aligner, f, protocol=cPickle.HIGHEST_PROTOCOL)                    
        print "Saved clustering to %s" % clustering_out

    df = evaluate_performance(hp, aligner, gt_file, evaluation_method, q=q)
    if df_out is not None:
        df.to_pickle(df_out)    
        print "Saved df to %s" % df_out
    
    return df
    
def plot_density(exp_res, title, xlim=(0.7, 1.0), ylim=(0.8, 1.0)):
    training_dfs = []
    for item in exp_res:
        training_data, training_df, best_training_row, match_res = item
        training_dfs.append(training_df)
    combined = pd.concat(training_dfs, axis=0)
    combined = combined.reset_index(drop=True)
#     f, ax = plt.subplots(figsize=(6, 6))    
#     sns.kdeplot(combined.Rec, combined.Prec, ax=ax)
#     sns.rugplot(combined.Rec, ax=ax)
#     sns.rugplot(combined.Prec, vertical=True, ax=ax)    
#     ax.set_xlim([0.7, 1.0])
#     ax.set_ylim([0.7, 1.0])
    g = sns.JointGrid(x="Rec", y="Prec", data=combined, xlim=xlim, ylim=ylim)
    g = g.plot_joint(sns.kdeplot)
    g = g.plot_marginals(sns.kdeplot, shade=True)
    ax = g.ax_joint
    ax.set_xlabel('Rec', fontsize=36)
    ax.set_ylabel('Prec', fontsize=36)
    ax = g.ax_marg_x
    ax.set_title(title, fontsize=36)  
    
def get_training_rows(exp_res, matching, no_files):
    rows = []
    for i in range(len(exp_res)):
        item = exp_res[i]
        training_data, training_df, best_training_row, match_res = item
        best_training_row['no_files'] = no_files
        best_training_row['matching'] = matching
        best_training_row['iter'] = i
        rows.append(best_training_row)
    return rows
    
def get_testing_rows(exp_res, matching, no_files):
    rows = []
    for i in range(len(exp_res)):
        item = exp_res[i]
        training_data, training_df, best_training_row, match_res = item
        if matching == 'MWG':
            temp = match_res[0:2] + match_res[4:]
            testing_results = temp + (no_files, matching, i)
        else:
            testing_results = match_res  + (no_files, matching, i)            
        rows.append(testing_results)
    return rows
    
def plot_training_boxplot(MW, MWG, cluster_match):
    rows = []
    rows1 = get_training_rows(MW, 'MW', 2)
    rows2 = get_training_rows(MWG, 'MWG', 2)
    rows3 = get_training_rows(cluster_match, 'Cluster-Match', 2)
    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    df3 = pd.DataFrame(rows3)    
    rows.extend(rows1)
    rows.extend(rows2)
    rows.extend(rows3)
    df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    ax = sns.boxplot(x="matching", y="F1", data=df, palette="Set3", width=0.5)
    ax.set_title('Training Performance', fontsize=36)
    return df1, df2, df3
    
def plot_testing_boxplot(MW, MWG, cluster_match):
    rows = []
    rows1 = get_testing_rows(MW, 'MW', 2)
    rows2 = get_testing_rows(MWG, 'MWG', 2)
    rows3 = get_testing_rows(cluster_match, 'Cluster-Match', 2)
    df1 = pd.DataFrame(rows1, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold', 'no_files', 'matching', 'iter'])
    df2 = pd.DataFrame(rows2, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold', 'no_files', 'matching', 'iter'])
    df3 = pd.DataFrame(rows3, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold', 'no_files', 'matching', 'iter'])    
    rows.extend(rows1)
    rows.extend(rows2)
    rows.extend(rows3)
    df = pd.DataFrame(rows, columns=['mass_tol', 'rt_tol', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'Threshold', 'no_files', 'matching', 'iter'])
    df = df.reset_index(drop=True)
    ax = sns.boxplot(x="matching", y="F1", data=df, palette="Set3", width=0.5)
    ax.set_title('Testing Performance', fontsize=36)
    ax.set_ylim([0.8, 1.0])
    return df1, df2, df3
    
def plot_scatter(exp_res, idx, df, title):
    item = exp_res[idx]
    training_data, training_df, best_training_row, match_res = item
    training_df = training_df.reset_index(drop=True)
    g = sns.JointGrid(x="Rec", y="Prec", data=training_df)
    g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")
    plt.figure(g.fig.number)
    plt.plot(df.Rec, df.Prec, '.r-')    
    # g = g.plot_marginals(sns.distplot, kde=False, color=".5")  
    g = g.plot_marginals(sns.kdeplot, shade=True)
    ax = g.ax_joint
    ax.set_xlabel('Rec')
    ax.set_ylabel('Prec')
    ax.set_ylim([0.7, 1.0])
    ax = g.ax_marg_x
    ax.set_title(title)
