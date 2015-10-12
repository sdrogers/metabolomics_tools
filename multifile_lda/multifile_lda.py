import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from numpy import int32
from numpy.random import RandomState
import sys
import pylab as plt

import multifile_utils as utils

class MultifileLDA(object):

    def __init__(self, random_state=None):
        
        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state    

        self.blowup = 10
    
    def _load_data(self, f, fragment_filename, neutral_loss_filename, ms1_filename, ms2_filename):
    
        print "Loading file " + str(f),
        fragment_data = pd.read_csv(fragment_filename, index_col=0)
        neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
        
        ms1 = pd.read_csv(ms1_filename, index_col=0)
        ms2 = pd.read_csv(ms2_filename, index_col=0)        
        ms2['fragment_bin_id'] = ms2['fragment_bin_id'].astype(str)
        ms2['loss_bin_id'] = ms2['loss_bin_id'].astype(str)
    
        data = pd.DataFrame()
    
        # discretise the fragment and neutral loss intensities values by converting it to 0 .. blowup
        fragment_data *= self.blowup
        data = data.append(fragment_data)
        neutral_loss_data *= self.blowup
        data = data.append(neutral_loss_data)
                
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
    
    def load_all(self, input_set):

        self.F = len(input_set)
        self.dfs = {}
        self.ms1s = {}
        self.ms2s = {}
        self.Ds = {}
        self.vocab = None

        for f in range(self.F):        
            
            entry = input_set[f]
            
            fragment_filename, neutral_loss_filename, ms1_filename, ms2_filename = entry
            df, vocab, ms1, ms2 = self._load_data(f, fragment_filename, neutral_loss_filename, ms1_filename, ms2_filename)
            nrow, ncol = df.shape
            assert nrow == len(ms1)
            assert ncol == len(vocab)

            self.Ds[f] = nrow
            self.dfs[f] = df
            self.ms1s[f] = ms1
            self.ms2s[f] = ms2
            self.vocab = vocab
            
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
                
    def print_top_words(self, n_words=10):
        for i, topic_dist in enumerate(self.topic_word_):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-n_words:-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))        
                
    def plot_e_alphas(self):
        plt.figure()
        ax = None
        for f in range(self.F):
            if ax is None:
                ax = plt.subplot(1, self.F, f+1)
            else:
                plt.subplot(1, self.F, f+1, sharey=ax)                
            post_alpha = self.posterior_alphas[f]
            e_alpha = post_alpha / np.sum(post_alpha)
            print e_alpha
            ind = range(len(e_alpha))
            plt.bar(ind, e_alpha, 0.5)        
            plt.title('File ' + str(f))
        plt.suptitle('Topic prevalence')
        plt.show()
        
    def plot_topic_word(self):
        plt.figure()
        plt.matshow(self.topic_word_)
        plt.colorbar()
        plt.title('Topic - word')
        plt.show()

    def plot_doc_topic(self):
        for doc_topic in self.doc_topic_:
            plt.figure()
            plt.matshow(doc_topic)
            plt.colorbar()
            plt.title('Doc - topic')
            plt.show()
            
def main():    
    
    lda = MultifileLDA()
    input_set = [
                 ('input/beer3pos_fragments_1.csv', 'input/beer3pos_losses_1.csv', 'input/beer3pos_ms1_1.csv','input/beer3pos_ms2_1.csv'),
                 ('input/beer3pos_fragments_2.csv', 'input/beer3pos_losses_2.csv', 'input/beer3pos_ms1_2.csv','input/beer3pos_ms2_2.csv'),
                 ('input/beer3pos_fragments_3.csv', 'input/beer3pos_losses_3.csv', 'input/beer3pos_ms1_3.csv','input/beer3pos_ms2_3.csv')
                 ]
    lda.load_all(input_set)    
    lda.run(30, 0.01, 0.1, n_burn=0, n_samples=20, n_thin=1)
    lda.plot_e_alphas()
    
if __name__ == "__main__": main()