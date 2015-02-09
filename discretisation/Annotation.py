from models import MassBin, IntervalTree
import numpy as np


class MolAnnotator:
    def annotate_mols(self, moldb, precursor_masses, mass_tol):
        """ A simple annotation experiment to see if we lose anything by binning:
        i. Take the M+H precursor mass from a standard file, match them against database within tolerance
        and see how many you get --> gold standard
        ii. Compare this with the discrete version
        """
            
        # the old-fashioned way of annotation in the continuous space
        print 'Checking continuous molecule annotations'
        unambiguous = set()
        ambiguous = set()
        for db_entry in moldb:
            found = 0
            for pc in precursor_masses:
                if self.mass_match(db_entry.mass, pc, mass_tol):
                    found = found+1
            if found==1:
                unambiguous.add(db_entry)
            elif found>1:
                ambiguous.add(db_entry)
        continuous_hits = len(unambiguous) + len(ambiguous)
        print '\tcontinuous_hits=' + str(continuous_hits) + '/' + str(len(moldb)) + ' molecules'
        print '\tambiguous=' + str(len(ambiguous)) + ' unambiguous=' + str(len(unambiguous))

        # now check if we lose anything by going discrete
        print 'Checking discrete molecule annotations'
        # first, make the bins
        lower, upper = self.bin_range(precursor_masses, mass_tol)
        the_bins = []
        for i in np.arange(len(precursor_masses)):
            low = lower[i]
            up = upper[i]
            the_bins.append(MassBin(low, up))        

        # count the hits
        T = IntervalTree(the_bins) # store bins in an interval tree
        unambiguous = set()
        ambiguous = set()
        for db_entry in moldb:
            matching_bins = T.search(db_entry.mass)
            if (len(matching_bins)==1): # exactly one matching bin
                unambiguous.add(db_entry)
            elif (len(matching_bins)>1): # more than one possible matching bins
                ambiguous.add(db_entry)
        discrete_hits = len(unambiguous) + len(ambiguous)
        print '\tdiscrete_hits=' + str(discrete_hits) + '/' + str(len(moldb)) + ' molecules'
        print '\tambiguous=' + str(len(ambiguous)) + ' unambiguous=' + str(len(unambiguous))    
        
    def mass_match(self, m1, m2, tol):
        return np.abs((m1-m2)/(m1))<tol*1e-6
        
    def bin_range(self, m1, tol):
        interval = m1*tol*1e-6
        upper = m1+interval
        lower = m1-interval
        return lower, upper
