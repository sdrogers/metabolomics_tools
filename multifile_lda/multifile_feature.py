import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from numpy import int32
from Queue import PriorityQueue


class MultifileFeatureExtractor(object):

    def __init__(self, input_set):

        self.all_ms1 = []
        self.all_ms2 = []
        self.all_dfs = []
        self.vocab = []
        
        self.common_fragments = []
        
        # load all the ms1 and ms2 files
        parent_masses = []
        for ms1_filename, ms2_filename in input_set:
            
            print "Loading %s" % ms1_filename
            ms1 = pd.read_csv(ms1_filename, index_col=0)

            print "Loading %s" % ms2_filename
            ms2 = pd.read_csv(ms2_filename, index_col=0)                    
            ms2.drop(['fragment_bin_id', 'loss_bin_id'], inplace=True, axis=1)
            
            self.all_ms1.append(ms1)
            self.all_ms2.append(ms2)
            
            frags_list = ms2['mz'].values.tolist()
            self.common_fragments.extend(frags_list)

#             # TODO: vectorise this!
#             for index, row in ms2.iterrows():
#                 parent_id = int(row['MSnParentPeakID'])
#                 parent_row = ms1.loc[ms1['peakID']==parent_id]
#                 parent_mz = parent_row[['mz']].values.flatten()[0]
#                 parent_masses.append(parent_mz)

#         losses = np.array(parent_masses) - np.array(self.common_fragments)
#         self.common_losses = losses.tolist()

        self.common_fragments = sorted(self.common_fragments)
#         self.common_losses = sorted(self.common_losses)

        self.F = len(input_set)
        print "Fragments = %d words" % len(self.common_fragments)
#         print "Losses = %d words" % len(self.common_losses)
                
    def _make_queue(self, all_ms2):
        q = PriorityQueue()     
        for f in range(self.F): 
            ms2 = all_ms2[f]  
            for index, row in ms2.iterrows():
                row_mz = row[['mz']].values.flatten()[0]
                q.put((row_mz, row, f)) 
        return q                
                
    def create_features(self):
        
        grouping_tol = 7
        q = self._make_queue(self.all_ms2)

        for mz in self.common_fragments:
            
            # calculate mz window
            max_ppm = mz * grouping_tol * 1e-06
            lower = mz - max_ppm
            upper = mz + max_ppm                
            
        
    def get_entry(self, f):
        
        df = self.all_dfs[f]
        vocab = self.vocab
        ms1 = self.all_ms1[f]
        ms2 = self.all_ms2[f]
        
        return df, vocab, ms1, ms2
