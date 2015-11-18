import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from numpy import int32
from numpy.random import RandomState
import sys
import pylab as plt

import cPickle
import gzip
import os
import re
import sys
import time
import timeit

import seaborn as sns

import multifile_utils as utils

class MultifileLDA(object):

    def __init__(self, random_state=None):
        
        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state    

    def load_all(self, input_set, scaling_factor=100, normalise=0):

        self.F = len(input_set)
        self.dfs = {}
        self.ms1s = {}
        self.ms2s = {}
        self.Ds = {}
        self.vocab = None

        for f in range(self.F):        
            
            entry = input_set[f]
            
            fragment_filename, neutral_loss_filename, ms1_filename, ms2_filename = entry
            df, vocab, ms1, ms2 = self._load_data(f, fragment_filename, neutral_loss_filename, 
                                                  ms1_filename, ms2_filename, 
                                                  scaling_factor, normalise)
            nrow, ncol = df.shape
            assert nrow == len(ms1)
            assert ncol == len(vocab)

            self.Ds[f] = nrow
            self.dfs[f] = df
            self.ms1s[f] = ms1
            self.ms2s[f] = ms2
            self.vocab = vocab
            
    def _load_data(self, f, fragment_filename, neutral_loss_filename, ms1_filename, ms2_filename, scaling_factor, normalise):
    
        print "Loading file " + str(f),
        fragment_data = pd.read_csv(fragment_filename, index_col=0)
        neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
        
        ms1 = pd.read_csv(ms1_filename, index_col=0)
        ms2 = pd.read_csv(ms2_filename, index_col=0)        
        ms2['fragment_bin_id'] = ms2['fragment_bin_id'].astype(str)
        ms2['loss_bin_id'] = ms2['loss_bin_id'].astype(str)

        # discretise the fragment and neutral loss intensities values
        if normalise == 1:
            data = self._normalise_1(scaling_factor, fragment_data, neutral_loss_data)
        elif normalise == 2:
            data = self._normalise_2(scaling_factor, fragment_data, neutral_loss_data, ms1)
                
        # get rid of NaNs, transpose the data and floor it
        data = data.replace(np.nan,0)
        data = data.transpose()
        sd = coo_matrix(data)
        sd = sd.floor()  
        npdata = np.array(sd.todense(), dtype='int32')
        print "data shape " + str(npdata.shape)
        df = pd.DataFrame(npdata)
        df.columns = data.columns
        df.index = data.index
    
        # vocab is just a string of the column names
        vocab = data.columns.values
                
        return df, vocab, ms1, ms2

    def _normalise_1(self, scaling_factor, fragment_data, neutral_loss_data):
        
        # converting it to 0 .. scaling_factor
        fragment_data *= scaling_factor
        neutral_loss_data *= scaling_factor
        data = pd.DataFrame()
        data = data.append(fragment_data)
        data = data.append(neutral_loss_data)
        return data
    
    def _normalise_2(self, scaling_factor, fragment_data, neutral_loss_data, ms1):

        # same as method 0, but with additional normalisation by of the parent MS1 intensities ratio
        fragment_data *= scaling_factor
        neutral_loss_data *= scaling_factor
        data = pd.DataFrame()
        data = data.append(fragment_data)
        data = data.append(neutral_loss_data)        
        
        intensities = ms1['intensity'].values
        intensity_ratios = intensities/np.max(intensities)
        n = 0
        for col in data.columns:
            data[col] *= intensity_ratios[n]
            n += 1
        
        return data            
            
    def run(self, K, alpha, beta, n_burn=100, n_samples=200, n_thin=0):
        
        self.K = K       
        self.N = len(self.vocab)     

        # beta is shared across all files
        self.beta = np.ones(self.N) * beta            
        
        # set the matrices for each file
        self.alphas = {}
        self.ckn = {}
        self.ck = {}
        self.cdk = {}
        self.cd = {}
        for f in range(self.F):
            self.alphas[f] = np.ones(self.K) * alpha
            self.ckn[f] = np.zeros((self.K, self.N), dtype=int32)
            self.ck[f] = np.zeros(self.K, dtype=int32)        
            self.cdk[f] = np.zeros((self.Ds[f], self.K), int32)
            self.cd[f] = np.zeros(self.Ds[f], int32)

        # randomly assign words to topics
        # also turn word counts in the document into a vector of word occurences
        print "Initialising "        
        self.Z = {}        
        self.document_indices = {}
        for f in range(self.F):
            print " - file " + str(f) + " ",
            for d in range(self.Ds[f]):
                if d%10==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                document = self.dfs[f].iloc[[d]]
                word_idx = utils.word_indices(document)
                word_locs = []
                for pos, n in enumerate(word_idx):
                    k = self.random_state.randint(self.K)
                    file_cdk = self.cdk[f]
                    file_cd = self.cd[f]
                    file_ckn = self.ckn[f]
                    file_ck = self.ck[f]
                    file_cdk[d, k] += 1
                    file_cd[d] += 1
                    file_ckn[k, n] += 1
                    file_ck[k] += 1
                    self.Z[(f, d, pos)] = k
                    word_locs.append((pos, n))
                self.document_indices[(f, d)] = word_locs
            print
        print
                
        # select the sampler function to use
        sampler_func = None
        try:
            # print "Using Numba for multi-file LDA sampling"
            # from multifile_cgs_numba import sample_numba
            # sampler_func = sample_numba
            print "Using Numpy for multi-file LDA sampling"
            from multifile_cgs_numpy import sample_numpy
            sampler_func = sample_numpy
        except Exception:
            print "Using Numpy for multi-file LDA sampling"
            from multifile_cgs_numpy import sample_numpy
            sampler_func = sample_numpy            

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.posterior_alphas = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.F, self.Ds, self.N, self.K, self.document_indices,
                self.alphas, self.beta, self.Z,
                self.cdk, self.cd, self.ckn, self.ck)
        
    def do_thresholding(self, th_doc_topic=0.05, th_topic_word=0.1):
 
        # save the thresholding values used for visualisation later
        self.th_doc_topic = th_doc_topic
        self.th_topic_word = th_topic_word
                     
        # get rid of small values in the matrices of the results
        # if epsilon > 0, then the specified value will be used for thresholding
        # otherwise, the smallest value for each row in the matrix is used instead
        self.thresholded_topic_word = utils.threshold_matrix(self.topic_word_, epsilon=th_topic_word)
        self.thresholded_doc_topic = []
        for f in range(len(self.doc_topic_)):
            self.thresholded_doc_topic.append(utils.threshold_matrix(self.doc_topic_[f], epsilon=th_doc_topic))        
                
    def print_top_words(self, with_probabilities=True, query=None):
        
        for i, topic_dist in enumerate(self.thresholded_topic_word):
            
            ordering = np.argsort(topic_dist)
            topic_words = np.array(self.vocab)[ordering][::-1]
            dist = topic_dist[ordering][::-1]        
            topic_name = 'Mass2Motif {}:'.format(i)
            
            print topic_name,                    
            for j in range(len(topic_words)):
                if dist[j] > 0:
                    if with_probabilities:
                        print '%s (%.3f),' % (topic_words[j], dist[j]),
                    else:
                        print('{},'.format(topic_words[j])),                            
                else:
                    break
            print
            print
            
    def plot_motif_degrees(self, interesting=None):
        
        if interesting is None:
            interesting = [k for k in range(self.K)]            

        file_ids = []
        topic_ids = []
        degrees = []        
        for f in range(self.F):

            file_ids.extend([f for k in range(self.K)])
            topic_ids.extend([k for k in range(self.K)])

            doc_topic = self.thresholded_doc_topic[f]
            columns = (doc_topic>0).sum(0)
            assert len(columns) == self.K
            degrees.extend(columns)

        rows = []
        for i in range(len(topic_ids)):            
            topic_id = topic_ids[i]
            if topic_id in interesting:
                rows.append((file_ids[i], topic_id, degrees[i]))

        df = pd.DataFrame(rows, columns=['file', 'M2M', 'degree'])
        sns.barplot(x="M2M", y="degree", hue='file', data=df)
                
        return df
                
    def plot_e_alphas(self, interesting=None):

        if interesting is None:
            interesting = [k for k in range(self.K)]            

        file_ids = []
        topic_ids = []
        alphas = []        
        for f in range(self.F):

            file_ids.extend([f for k in range(self.K)])
            topic_ids.extend([k for k in range(self.K)])

            post_alpha = self.posterior_alphas[f]
            e_alpha = post_alpha / np.sum(post_alpha)
            assert len(e_alpha) == self.K
            alphas.extend(e_alpha.tolist())

        rows = []
        for i in range(len(topic_ids)):            
            topic_id = topic_ids[i]
            if topic_id in interesting:
                rows.append((file_ids[i], topic_id, alphas[i]))

        df = pd.DataFrame(rows, columns=['file', 'M2M', 'alpha'])
        sns.barplot(x="M2M", y="alpha", hue='file', data=df)
                
        return df
            
    @classmethod
    def resume_from(cls, project_in):
        start = timeit.default_timer()        
        with gzip.GzipFile(project_in, 'rb') as f:
            obj = cPickle.load(f)
            stop = timeit.default_timer()
            print "Project loaded from " + project_in + " time taken = " + str(stop-start)
            return obj  
         
    def save_project(self, project_out, message=None):
        start = timeit.default_timer()        
        self.last_saved_timestamp = str(time.strftime("%c"))
        self.message = message
        with gzip.GzipFile(project_out, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            stop = timeit.default_timer()
            print "Project saved to " + project_out + " time taken = " + str(stop-start)    
                          
def main():    
    
    lda = MultifileLDA()
    input_set = [
                 ('input/beer3pos_fragments_1.csv', 'input/beer3pos_losses_1.csv', 'input/beer3pos_ms1_1.csv','input/beer3pos_ms2_1.csv'),
                 ('input/beer3pos_fragments_2.csv', 'input/beer3pos_losses_2.csv', 'input/beer3pos_ms1_2.csv','input/beer3pos_ms2_2.csv'),
                 ('input/beer3pos_fragments_3.csv', 'input/beer3pos_losses_3.csv', 'input/beer3pos_ms1_3.csv','input/beer3pos_ms2_3.csv')
                 ]
    lda.load_all(input_set)    
    lda.run(300, 0.01, 0.1, n_burn=0, n_samples=20, n_thin=1)
    lda.plot_e_alphas()
    
if __name__ == "__main__": main()