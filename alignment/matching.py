import math
import time
import sys
import itertools

import numpy as np
import scipy.sparse
from scipy import weave
from scipy.sparse import *

import networkx as nx
from networkx.algorithms import matching

from models import *
from utils import *
from clustering import *

from pq import PriorityQueue
from IntervalTree import IntervalTree
    
class MaxWeightedMatching:
        
    def __init__(self, men, women, options):
        
        self.men = men
        self.women = women
        self.options = options
        self.exact_match = self.options.exact_match
        self.grouping_method = self.options.grouping_method
        self.dmz = float(self.options.dmz)
        self.drt = float(self.options.drt)
        self.grt = float(self.options.grt)
        self.alpha = float(self.options.alpha)      
        self.num_samples = int(self.options.num_samples)
        self.burn_in = int(self.options.burn_in)
        self.skip_matching = self.options.skip_matching
        self.verbose = self.options.verbose

    def do_matching(self):
        
        if len(self.men.rows) == 0:

            # for first time matching 
            matched_results = AlignmentFile(self.women.filename, self.verbose)
            matched_results.parent_dir = self.women.parent_dir   
            matched_results.add_rows(self.women.rows)         
            return matched_results
        
        else:
            
            # for subsequent merging
            matched_results = AlignmentFile(self.men.filename + "_" + self.women.filename, self.verbose)

            # do clustering only
            if self.skip_matching:
                clusterer = self.get_clusterer(self.men, self.options)
                A = clusterer.do_clustering()
                clusterer = self.get_clusterer(self.women, self.options)
                B = clusterer.do_clustering()
                return matched_results

            # compute distance matrix            
            print "Computing score matrix"
            score_arr, Q = self.compute_scores(self.men, self.women, self.dmz, self.drt)

            # combine scores, if necessary
            if self.options.use_group:
                print "\nCombining grouping information"
                clusterer = self.get_clusterer(self.men, self.options)
                A = clusterer.do_clustering()
                clusterer = self.get_clusterer(self.women, self.options)
                B = clusterer.do_clustering()
                score_arr = self.combine_scores(score_arr, A, B, Q)
            
            # do approximate or exact matching here
            print "Running matching"
            if self.options.exact_match:
                mate = self.hungarian_matching(self.men, self.women, score_arr)
            else:
                # mate = self.approximate_matching_scipy(self.men, self.women, score_arr)
                mate = self.approximate_matching_pq(self.men, self.women, score_arr)
            print

            # process the matched rows
            row_id = 0
            for key, value in mate.iteritems():
                # doesn't matter who's key or value
                man = key
                woman = value
                matched_row = AlignmentRow(row_id)
                matched_row.add_features(man)
                matched_row.add_features(woman)
                man.aligned = True
                woman.aligned = True
                row_id = row_id + 1
                matched_results.add_row(matched_row)                
                
            # then process all the unmatched rows
            unaligned_rows = self.men.get_unaligned_rows()
            unaligned_rows.extend(self.women.get_unaligned_rows())
            matched_results.add_rows(unaligned_rows)
            for row in unaligned_rows:
                row.aligned = True

            return matched_results
        
    def get_clusterer(self, alignment_file, options):
        if self.options.grouping_method.lower() == "greedy":
            clusterer = GreedyClustering(alignment_file, options)
        elif self.options.grouping_method.lower() == "posterior":
            clusterer = MixtureModelClustering(alignment_file, options)
        else:
            print "Unknown grouping method " + options.grouping_method
            exit(1)
        return clusterer
        
    def compute_dist(self, row1, row2, dmz, drt):
        
        mass1 = row1.get_average_mass()
        mass2 = row2.get_average_mass()
        rt1 = row1.get_average_rt()
        rt2 = row2.get_average_rt()
        
        rt = rt1 - rt2
        mz = mass1 - mass2
        dist = math.sqrt((rt*rt)/(drt*drt) + (mz*mz)/(dmz*dmz))
        
        return dist
    
    def compute_scores(self, men, women, dmz, drt):
        
        # construct a score matrix
        n_row = len(men.rows)
        n_col = len(women.rows)
        dist_arr = lil_matrix((n_row, n_col))
        max_dist = 0        
        
        T = IntervalTree(women.rows)
        for i in range(n_row):
            man = men.rows[i]
            mass_lower, mass_upper = man.get_mass_range(dmz)
            candidate_women = T.search(int(mass_lower), int(mass_upper))
            for woman in candidate_women:
                if man.is_within_tolerance(woman, dmz, drt):
                    dist = self.compute_dist(man, woman, dmz, drt)
                    j = woman.row_id
                    dist_arr[i, j] = dist
                    if dist > max_dist:
                        max_dist = dist

        # make this into a score matrix
        dist_arr = dist_arr.tocoo()
        score_arr = lil_matrix((n_row, n_col))
        Q = lil_matrix((n_row, n_col))
        max_score = 0
        # see http://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        for i, j, v in itertools.izip(dist_arr.row, dist_arr.col, dist_arr.data):
            score = 1-(v/max_dist)
            score_arr[i, j] = score
            Q[i, j] = 1
            if score > max_score:
                max_score = score
                                  
        # normalise
        score_arr = score_arr * (1/max_score)
        return score_arr, Q          
        
    def combine_scores(self, W, A, B, Q):

        print " - Combining scores"
        sys.stdout.flush()

        W_row, W_col = W.shape
        A_row, A_col = A.shape
        B_row, B_col = B.shape        

        # turn to CSR for faster computation
        A = A.tocsr()
        B = B.tocsr()
        W = W.tocsr()
        
        # delete the diagonal entries for A and B, see 
        # http://stackoverflow.com/questions/22660374/remove-set-the-non-zero-diagonal-elements-of-a-sparse-matrix-in-scipy
        A = A - scipy.sparse.dia_matrix((A.diagonal()[scipy.newaxis, :], [0]), shape=(A_row, A_row))
        B = B - scipy.sparse.dia_matrix((B.diagonal()[scipy.newaxis, :], [0]), shape=(B_row, B_row))
        
        # do the multiplication to upweight / downweight
        print "\tComputing D=(AW)"
        AW = A * W
        print "\tComputing D=(AW)B"
        AWB = AW * B
        
        # mask the resulting output
        print "\tComputing D.*Q"                    
        D = AWB.multiply(Q)
        
        # normalise it
        max_score = D.max()
        D = D * (1/max_score)

        # combine with original scores
        print "\tComputing W'=(alpha.*W)+((1-alpha).*D)"            
        W = W * self.alpha
        D = D * (1-self.alpha)
        score_arr = W + D
        max_score = score_arr.max()
        score_arr = score_arr * (1/max_score)
                
        return score_arr
    
    def hungarian_matching(self, men, women, score_arr):

        # make graph
        score_arr = score_arr.tocoo()
        G = nx.Graph()
        for i, j, v in itertools.izip(score_arr.row, score_arr.col, score_arr.data):
            man = self.men.rows[i]
            woman = self.women.rows[j]                
            G.add_edge(man, woman, weight=v)
            
        mate = matching.max_weight_matching(G, maxcardinality=True)
        
        return mate
    
    def approximate_matching_networkx(self, men, women, score_arr):

        # make graph
        score_arr = score_arr.tocoo()
        G = nx.Graph()
        for i, j, v in itertools.izip(score_arr.row, score_arr.col, score_arr.data):
            man = self.men.rows[i]
            woman = self.women.rows[j]                
            G.add_edge(man, woman, weight=v)

        M = {}
        total = G.number_of_edges()
        tick = total / 20
        if tick == 0:
            tick = total
                
        # while there's an edge
        i = G.number_of_edges()
        while i > 0:
            
            # find the heaviest edge in graph
            sorted_edges = sorted(G.edges(data=True), key=lambda (source,target,data): data['weight'], reverse=True)
            heaviest_edge = sorted_edges[0]
            
            # add e to M
            key_row = heaviest_edge[0]
            value_row = heaviest_edge[1]
            M[key_row] = value_row
            
            # remove e and all edges adjacent to e from graph
            # G.remove_node(key_row)
            # G.remove_node(value_row)
            neighbours = G.neighbors(key_row)
            for neighbour in neighbours:
                G.remove_edge(key_row, neighbour)
            neighbours = G.neighbors(value_row)
            for neighbour in neighbours:
                G.remove_edge(value_row, neighbour)
            
        return M
                        
    def approximate_matching_pq(self, men, women, score_arr):
        ''' Faster version using priority queue '''
        score_arr = score_arr.tolil()
        mate = {}
        men = list(men.rows) # copy them first
        women = list(women.rows)

        # make the queue first
        q = self.make_queue(score_arr)
        total = q.size
        tick = total / 20
        if tick == 0:
            tick = total
        
        while not q.empty():

            pq_entry = q.get()
            priority = pq_entry[0]
            item = pq_entry[1]
            i = item[0]
            j = item[1]   
            
            if score_arr[i, j] != 0:
                man = men[i]
                woman = women[j]
                mate[man] = woman                
                score_arr[i, :] = 0
                score_arr[:, j] = 0
        
        return mate
    
    def make_queue(self, score_arr):
        q = PriorityQueue()        
        score_arr = score_arr.tocoo()
        for i, j, v in itertools.izip(score_arr.row, score_arr.col, score_arr.data):
            inverted_score = -v
            item = (i, j)
            q.put((inverted_score, item)) 
        return q
