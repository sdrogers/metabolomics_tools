from Queue import PriorityQueue
import sys

from numpy import int32
from scipy.sparse import coo_matrix

import numpy as np
import pandas as pd


class MultifileFeatureExtractor(object):

    def __init__(self, input_set, fragment_grouping_tol, loss_grouping_tol, loss_threshold_min_count, loss_threshold_max_val):

        self.all_ms1 = []
        self.all_ms2 = []
        self.all_dfs = []
        self.vocab = []
        self.fragment_grouping_tol = fragment_grouping_tol
        self.loss_grouping_tol = loss_grouping_tol
        self.loss_threshold_min_count = loss_threshold_min_count
        self.loss_threshold_max_val = loss_threshold_max_val
        
        # load all the ms1 and ms2 files
        self.F = len(input_set)        
        for ms1_filename, ms2_filename in input_set:
            
            print "Loading %s" % ms1_filename
            ms1 = pd.read_csv(ms1_filename, index_col=0)
            self.all_ms1.append(ms1)

            print "Loading %s" % ms2_filename
            ms2 = pd.read_csv(ms2_filename, index_col=0)                    
            ms2.drop(['fragment_bin_id', 'loss_bin_id'], inplace=True, axis=1)            
            self.all_ms2.append(ms2)
            
    def make_fragment_queue(self):
        q = PriorityQueue()     
        for f in range(self.F): 
            print "Processing fragments for file %d" % f
            ms2 = self.all_ms2[f]  
            for index, row in ms2.iterrows():
                fragment_mz = row['mz']
                fragment_id = row['peakID']
                item = (fragment_mz, fragment_id, f, row)
                q.put(item) 
        return q                

    def make_loss_queue(self):
        q = PriorityQueue()     
        for f in range(self.F): 
            print "Processing losses for file %d" % f
            ms1 = self.all_ms1[f]
            ms2 = self.all_ms2[f]  
            for index, row in ms2.iterrows():
                parent_id = row['MSnParentPeakID']                    
                parent_row = ms1.loc[ms1['peakID']==parent_id]                
                row_mz = row['mz']
                row_id = row['peakID']
                parent_mz = parent_row['mz']
                loss_mz = parent_mz - row_mz
                loss_mz = loss_mz.values[0]
                item = (loss_mz, row_id, f, row)
                q.put(item) 
        return q            
    
    def group_features(self, q, grouping_tol, check_threshold=False):
        
        total_ms2 = len(q.queue)

        groups = {}
        k = 0
        group = []
        unique_check = []
        while not q.empty():
            
            current_item = q.get()
            current_mass = current_item[0]
            current_id = current_item[1]
            current_file = current_item[2]
            current_row = current_item[3]
            item = (current_row, current_file, current_mass)
            group.append(item)

            # check if the next mass is outside tolerance            
            if len(q.queue) > 0:
                head = q.queue[0]
                head_mass = head[0]
                _, upper = self._mass_range(current_mass, grouping_tol)                            
                if head_mass > upper:
                    
                    # check if the current group is valid before starting a new group
#                     unique_check = set()
#                     for row, f, val in group:
#                         this_parent_id = row['MSnParentPeakID']
#                         this_file_id = f
#                         key = (this_file_id, this_parent_id)
#                         if key in unique_check:
#                             self._print_group(group)
#                             msg = "Duplicate feature from (file %d, parent %d) already in the bin, maybe change the threshold?" % key
#                             raise ValueError(msg)
#                         unique_check.add(key)

                    if check_threshold:
                        valid = True
                        if len(group) < self.loss_threshold_min_count:
                            # print "len(group) = %d < self.loss_threshold_min_count %d" % (len(group), self.loss_threshold_min_count)
                            valid = False
                        if current_mass > self.loss_threshold_max_val:
                            # print "current_mass %.5f > self.loss_threshold_max_val %f" % (current_mass, self.loss_threshold_max_val)
                            valid = False
                        if valid:
                            groups[k] = group
                            k += 1
                        # else:
                            # print "Discard %d" % k
                    else: # nothing to check
                        groups[k] = group
                        k += 1
                    group = [] # whether valid or not, discard this group
            else:
                # no more item
                groups[k] = group

        K = len(groups)
        print "Total groups=%d" % K
        return groups    
                                
    def create_dataframes(self, fragment_groups, loss_groups):

        # initialise fragment vocab 
        fragment_group_words = self._generate_words(fragment_groups, 'fragment')
        loss_group_words = self._generate_words(loss_groups, 'loss')
        self.vocab.extend(fragment_group_words.values())
        self.vocab.extend(loss_group_words.values())

        # initialise the dataframes for each file
        for f in range(self.F):
            df = self._init_df(f, self.vocab)
            self.all_dfs.append(df)
            
        # populate the dataframes
        self._populate_df(fragment_groups, fragment_group_words)
        self._populate_df(loss_groups, loss_group_words)
                    
    def normalise(self, f, scaling_factor):

        df = self.all_dfs[f]
        
        df *= scaling_factor
        df = df.transpose()
        df = df.apply(np.floor)            
        print "file %d data shape %s" % (f, df.shape)
        self.all_dfs[f] = df             
        
        ms2 = self.all_ms2[f]
        ms2['fragment_bin_id'] = ms2['fragment_bin_id'].astype(str)
        ms2['loss_bin_id'] = ms2['loss_bin_id'].astype(str)        
                    
    def get_entry(self, f):
        df = self.all_dfs[f]
        vocab = self.vocab
        ms1 = self.all_ms1[f]
        ms2 = self.all_ms2[f]
        return df, vocab, ms1, ms2
    
    def _mass_range(self, mass_centre, mass_tol):
        interval = mass_centre * mass_tol * 1e-6
        mass_start = mass_centre - interval
        mass_end = mass_centre + interval
        return (mass_start, mass_end)          
                    
    def _get_doc_label(self, mz_val, rt_val, pid_val):
        mz = np.round(mz_val, 5)
        rt = np.round(rt_val, 3)
        doc_label = '%s_%s_%s' % (mz, rt, pid_val)
        return doc_label
    
    def _print_group(self, group):
        print "%d members in the group" % len(group)
        for row, f, val in group:
            this_parent_id = row['MSnParentPeakID']
            this_file_id = f
            this_peak_id = row['peakID']
            key = (this_file_id, this_parent_id, this_peak_id)
            print "- %d %d %d" % key
 
    def _generate_words(self, groups, prefix):
        group_words = {}
        for k in groups:
            group = groups[k]
            group_vals = []
            for row, f, val in group:
                group_vals.append(val)
            mean_mz = np.mean(np.array(group_vals))
            rounded_mz = np.round(mean_mz, 5)
            w = '%s_%s' % (prefix, rounded_mz)
            group_words[k] = w
        return group_words
    
    def _init_df(self, f, vocab):
        
        # generate column labels
        doc_labels = []
        ms1 = self.all_ms1[f]
        mzs = ms1['mz'].values
        rts = ms1['rt'].values
        pids = ms1['peakID'].values
        n_words = len(ms1)
        for n in range(n_words):
            doc_label = self._get_doc_label(mzs[n], rts[n], pids[n])
            doc_labels.append(doc_label)
        
        # generate index
        row_labels = vocab

        # create the df with row and col labels        
        df = pd.DataFrame(index=row_labels, columns=doc_labels)
        df = df.fillna(0) # fill with 0s rather than NaNs
        return df
    
    def _populate_df(self, groups, group_words):

        for k in groups:
            
            w = group_words[k]
            tokens = w.split('_')
            word_type = tokens[0]
            word_val = float(tokens[1])

            assert word_type == 'fragment' or word_type == 'loss'                
            if k % 100 == 0:
                print "Populating dataframe for %s group %d/%d" % (word_type, k, len(groups))
            
            group = groups[k]
            for row, f, val in group:

                ms1 = self.all_ms1[f]
                ms2 = self.all_ms2[f]
                df = self.all_dfs[f]

                # update bin column in the original ms2 row
                pos = (ms2['peakID']==row['peakID'])       
                bin_type = word_type + '_bin_id'         
                ms2.loc[pos, bin_type] = word_val
                
                # find the column label
                parent_row = ms1.loc[ ms1['peakID'] == row['MSnParentPeakID'] ]
                doc_label = self._get_doc_label(parent_row['mz'].values[0], 
                                               parent_row['rt'].values[0], 
                                               parent_row['peakID'].values[0])

                # update intensity value in the df
                df.loc[w, doc_label] = row['intensity']