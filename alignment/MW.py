#!/usr/bin/python

import sys
import getopt
import os
import glob
import argparse
import re

from models import AlignmentFile
from models import AlignmentRow
from models import Feature
from alignment import BaseAligner
from alignment import MaxWeightedAligner

def print_banner():
    '''Prints some banner at program start'''
    print "MW.py -- maximum-weighted matching alignment"
    
def get_options(argv):
    '''Parses command-line options'''
    parser = argparse.ArgumentParser()
    # core arguments
    parser.add_argument('-i', required=True, dest='input_dir', help='input directory')
    parser.add_argument('-o', required=True, dest='output_path', help='output path')
    parser.add_argument('-dmz', required=True, help='alignment m/z tolerance window')
    parser.add_argument('-drt', required=True, help='alignment RT tolerance window')
    parser.add_argument('-method', dest='alignment_method', help='alignment method. Choices are greedy, mw')
    parser.set_defaults(alignment_method='mw')
    parser.add_argument('-exact_match', dest='exact_match', action='store_true', help='perform exact match if using MW?')
    parser.set_defaults(exact_match=False)
    # grouping arguments
    parser.add_argument('-g', dest='use_group', action='store_true', help='use grouping information?')
    parser.set_defaults(use_group=False)
    parser.add_argument('-p', dest='use_peakshape', action='store_true', help='use peak shape correlations information?')
    parser.set_defaults(use_peakshape=False)    
    parser.add_argument('-mcs', help='minimum correlations of peak shapes signal.')
    parser.set_defaults(mcs=0.9)
    parser.add_argument('-gm', dest='grouping_method', help='grouping method. Choices are greedy, posterior')
    parser.set_defaults(grouping_method='posterior')
    parser.add_argument('-alpha', help='clustering weight ratio')
    parser.set_defaults(alpha=0.5)
    parser.add_argument('-grt', help='grouping RT tolerance window.')
    parser.set_defaults(grt=2)
    parser.add_argument('-dp_alpha', help='DP concentration parameter')
    parser.set_defaults(dp_alpha=1)
    parser.add_argument('-num_samples', help='no of samples after burn-in')
    parser.set_defaults(num_samples=100)
    parser.add_argument('-burn_in', help='no of samples during burn-in')
    parser.set_defaults(burn_in=100)
    parser.add_argument('-s', dest='skip_matching', action='store_true', help='skip the matching stage, only do clustering')
    parser.set_defaults(skip_matching=False)
    parser.add_argument('-always_recluster', dest='always_recluster', action='store_true', help='always initialise a new mixture model clustering instead of reading previously-cached results')
    parser.set_defaults(always_recluster=False)
    parser.add_argument('-v', dest='verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)

    # parse it
    options = parser.parse_args(argv)
    if options.verbose:
        print "Options", options
    return options
    
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)    
    
def read_files(input_dir, verbose):
    '''Reads all csv and txt files inside input_dir'''
    filelist = []
    # process only csv and txt files in input_dir
    types = ('*.csv', '*.txt')
    os.chdir(input_dir)
    for files in types:
        filelist.extend(glob.glob(files))
    filelist = natural_sort(filelist)
    # store the loaded data into alignment_files list
    alignment_files = []        
    for fileitem in filelist:
        full_path = os.path.join(input_dir, fileitem);
        alignment_file = AlignmentFile(full_path, verbose)
        alignment_file.get_data()
        alignment_files.append(alignment_file)
    return alignment_files

def do_alignment(options):
    '''Performs feature-based alignment'''
    # retrieve input files
    input_dir = options.input_dir
    verbose = options.verbose
    alignment_files = read_files(input_dir, verbose)    
    # pick alignment method
    alignment_method = options.alignment_method
    aligner = BaseAligner.select_aligner(alignment_method, alignment_files, options)
    # do the alignment and write the output
    aligner.do_alignment()
    output_path = options.output_path
    aligner.save_results(output_path)    

def main(argv):    
    print_banner()
    options = get_options(argv)
    do_alignment(options)

if __name__ == "__main__":
   main(sys.argv[1:])
