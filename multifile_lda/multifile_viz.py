import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import matplotlib.patches as mpatches
from random import uniform
from __builtin__ import False

class MultifileLDAViz(object):
    
    def __init__(self, lda):
        self.lda = lda    

    def plot_motif_degrees(self, interesting=None):
        
        if interesting is None:
            interesting = [k for k in range(self.lda.K)]            

        file_ids = []
        topic_ids = []
        degrees = []        
        for f in range(self.lda.F):

            file_ids.extend([f for k in range(self.lda.K)])
            topic_ids.extend([k for k in range(self.lda.K)])

            doc_topic = self.lda.thresholded_doc_topic[f]
            columns = (doc_topic>0).sum(0)
            assert len(columns) == self.lda.K
            degrees.extend(columns)

        rows = []
        for i in range(len(topic_ids)):            
            topic_id = topic_ids[i]
            if topic_id in interesting:
                rows.append((file_ids[i], topic_id, degrees[i]))

        df = pd.DataFrame(rows, columns=['file', 'M2M', 'degree'])
        sns.barplot(x="M2M", y="degree", hue='file', data=df)
                
        return df
    
    def get_y_pos(self, intensity):
        if intensity < 0.9:
            y_pos = intensity + 0.2 + uniform(0.0, 0.6)
        else:
            y_pos = intensity + 0.2 + 0.05
        return y_pos
    
    def find_nearest(self, value, array):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    
    def has_overlap(self, value, seen_before):
        if len(seen_before) == 0:
            return False
        nearest = self.find_nearest(value, np.array(seen_before))
        if np.abs(nearest-value)<0.05:
            return True
        else:
            return False
    
    def plot_docs(self, k, xlim_max=500):

        interesting = [k]
        topic_words_map = self.lda.get_top_words(with_probabilities=True, selected=interesting)
        topic_words, dist = topic_words_map[k]

        thresholded_topic_words = []
        for j in range(len(topic_words)):
            if dist[j] > 0:
                thresholded_topic_words.append(topic_words[j])

        for f in range(self.lda.F): # for each input file

            doc_topic = self.lda.thresholded_doc_topic[f]
            col = doc_topic[:, k]
            pos = np.nonzero(col)
            
            neutral_loss_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            parent_colour = 'red'
            fragment_colour = 'blue'
            loss_colour = 'green'
            other_colour = 'darkgray'
            
            df = self.lda.ms1s[f].iloc[pos]
            for index, row in df.iterrows(): # for every fragmentation spectrum

                parent_peakid = int(row['peakID'])
                ms2_rows = self.lda.ms2s[f].loc[self.lda.ms2s[f]['MSnParentPeakID'] == parent_peakid]                
                # display(ms2_rows)
                
                figsize=(10, 6)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                
                parent_mass = row['mz']
                parent_rt = row['rt']
                parent_intensity = 0.25                

                # plot all
                masses = ms2_rows['mz'].values
                intensities = ms2_rows['intensity'].values
                num_peaks = len(masses)
                for j in range(num_peaks):
                    mass = masses[j]
                    intensity = intensities[j]
                    plt.plot((mass, mass), (0, intensity), linewidth=2.0, color=other_colour)
                 
                # plot the fragments explained by the topic
                seen_before = []
                words = ms2_rows['fragment_bin_id'].values
                for j in range(num_peaks):
                    mass = masses[j]
                    intensity = intensities[j]
                    w = 'fragment_%s' % words[j]
                    if w in thresholded_topic_words:

                        fragment_str = w.split('_')[1]
                        fragment_str = "%.4f" % float(fragment_str)                        
                        plt.plot((mass, mass), (0, intensity), linewidth=3.0, color=fragment_colour)
                        
                        # try to avoid overlapping labels
                        x_pos = mass
                        y_pos = self.get_y_pos(intensity)
                        collision = self.has_overlap(y_pos, seen_before)
                        while collision:
                            y_pos = self.get_y_pos(intensity)
                            collision = self.has_overlap(y_pos, seen_before)
                            
                        ax.annotate(fragment_str, xy=(mass, intensity), xytext=(x_pos, y_pos),
                                    arrowprops=dict(facecolor=fragment_colour, arrowstyle='->'),
                                    weight='bold',
                                    horizontalalignment='center', verticalalignment='top', alpha=0.75)
                        
                # plot the losses explained by the topic
                neutral_loss_count = 0                
                words = ms2_rows['loss_bin_id'].values
                for j in range(num_peaks):
                    mass = masses[j]
                    intensity = intensities[j]
                    w = 'loss_%s' % words[j]
                    if w in thresholded_topic_words:

                        # draw the neutral loss arrow
                        arrow_x1 = parent_mass
                        arrow_y1 = neutral_loss_positions[neutral_loss_count]
                        arrow_x2 = (mass-parent_mass)+5
                        arrow_y2 = 0
                        neutral_loss_count += 1
                        plt.arrow(arrow_x1, arrow_y1, arrow_x2, arrow_y2, head_width=0.05, head_length=4.0, 
                                  width=0.005, fc=loss_colour, ec=loss_colour)
            
                        # draw neutral loss label
                        text_x = mass+(parent_mass-mass)/4
                        text_y = arrow_y1+0.025
                        loss_str = w.split('_')[1]
                        loss_str = "%.4f" % float(loss_str)
                        t = ax.text(text_x, text_y, loss_str, ha="left", va="center", rotation=0,
                                    color=loss_colour, fontweight='bold')

                # plot the parent last
                plt.plot((parent_mass, parent_mass), (0, parent_intensity), linewidth=3.0, color=parent_colour)
                plt.title('File %d MS1 peakid=%d mz=%.5f rt=%.5f' % (f, parent_peakid, parent_mass, parent_rt), y=1.08)
                plt.ylim([0, 1.2])
                if xlim_max is not None:
                    plt.xlim([0, xlim_max])
                        
                parent_patch = mpatches.Patch(color=parent_colour, label='Parent peak')
                other_patch = mpatches.Patch(color=other_colour, label='Fragment peaks')
                fragment_patch = mpatches.Patch(color=fragment_colour, label='Topic fragment')
                loss_patch = mpatches.Patch(color=loss_colour, label='Topic loss')                
                plt.legend(handles=[parent_patch, other_patch, fragment_patch, loss_patch], 
                           frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5),
                          fancybox=True, shadow=True)  
                plt.show()                        
                plt.close()
                                                                
    def plot_e_alphas(self, interesting=None):

        if interesting is None:
            interesting = [k for k in range(self.lda.K)]            

        file_ids = []
        topic_ids = []
        alphas = []        
        for f in range(self.lda.F):

            file_ids.extend([f for k in range(self.lda.K)])
            topic_ids.extend([k for k in range(self.lda.K)])

            post_alpha = self.lda.posterior_alphas[f]
            e_alpha = post_alpha / np.sum(post_alpha)
            assert len(e_alpha) == self.lda.K
            alphas.extend(e_alpha.tolist())

        rows = []
        for i in range(len(topic_ids)):            
            topic_id = topic_ids[i]
            if topic_id in interesting:
                rows.append((file_ids[i], topic_id, alphas[i]))

        df = pd.DataFrame(rows, columns=['file', 'M2M', 'alpha'])
        sns.barplot(x="M2M", y="alpha", hue='file', data=df)
                
        return df