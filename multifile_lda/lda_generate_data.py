""" Produces some synthetic data according to the generative process of LDA.
See example usage in the main method of lda_cgs.py
"""
from numpy import int64
from pandas.core.frame import DataFrame

import numpy as np
import pylab as plt
import pandas as pd


class LdaDataGenerator(object):
    
        def __init__(self, alpha, make_plot=False):
            self.alpha = alpha
            self.make_plot = make_plot

        def generate_word_dists(self, n_topics, vocab_size, document_length):
                        
            width = vocab_size/n_topics
            word_dists = np.zeros((n_topics, vocab_size))
     
            for k in range(n_topics):
                temp = np.zeros((n_topics, width))
                temp[k, :] = int(document_length / width)
                word_dists[k,:] = temp.flatten()
     
            word_dists /= word_dists.sum(axis=1)[:, np.newaxis] # turn counts into probabilities     
            if self.make_plot:
                self._plot_nicely(word_dists, 'Topic X Words', 'N', 'K')
            return word_dists              
        
        def generate_document(self, word_dists, n_topics, vocab_size, document_length):

            # sample topic proportions with uniform dirichlet parameter alpha of length n_topics
            theta = np.random.mtrand.dirichlet([self.alpha] * n_topics)

            # for every word in the vocab for this document
            d = np.zeros(vocab_size)
            for n in range(document_length):
            
                # sample a new topic index    
                k = np.random.multinomial(1, theta).argmax()
                
                # sample a new word from the word distribution of topic k
                w = np.random.multinomial(1, word_dists[k,:]).argmax()

                # increase the occurrence of word w in document d
                d[w] += 1

            return d
        
        def generate_input_df(self, n_topics, vocab_size, document_length, n_docs, n_copies):
                        
            print "Generating input DF"
                        
            # word_dists is the topic x document_length matrix
            word_dists = self.generate_word_dists(n_topics, vocab_size, document_length)                        

            dfs = []
            for j in range(n_copies):
            
                # generate each document x terms vector
                docs = np.zeros((vocab_size, n_docs), dtype=int64)
                for i in range(n_docs):
                    docs[:, i] = self.generate_document(word_dists, n_topics, vocab_size, document_length)
                                    
                df = DataFrame(docs)
                df = df.transpose()
                dfs.append(df)
                
                print df.shape            
                if self.make_plot:
                    self._plot_nicely(df, 'Documents_%d X Words' % j, 'Words', 'Documents')
            
            # generate the vocab
            vocab = []            
            for n in range(vocab_size):
                word = "word_" + str(n)
                vocab.append(word)
            vocab = np.array(vocab)            

            return dfs, vocab
        
        def generate_from_file(self, df_infile, vocab_infile):
            
            # read data frame
            df = pd.read_csv(df_infile, index_col=0)

            # here we need to change column type from string to integer for 
            # other parts in gibbs sampling to work ...
            # TODO: check why, because this means we cannot set the column 
            # names in the dataframe to the words!
            df.rename(columns = lambda x: int(x), inplace=True)
            
            vocab = np.genfromtxt(vocab_infile, dtype='str')
            return df, vocab
        
        def _plot_nicely(self, mat, title, xlabel, ylabel, outfile=None):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(b=False)            
            im = ax.matshow(mat, cmap=plt.cm.jet)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect(2)
            ax.set_aspect('auto')
            plt.colorbar(im)
            if outfile is not None:
                plt.savefig(outfile)
            plt.show()