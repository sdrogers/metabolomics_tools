from interval_tree import IntervalTree
import utils


class MolAnnotator:

    def identify_normal(self, moldb, precursor_masses, mass_tol):
        print "Checking normal identification"
        for db_entry in moldb:
            db_entry.set_ranges(mass_tol)
        found = 0
        unique = set()
        for pc in precursor_masses:
            hits = utils.db_hit(moldb, pc)
            found = found + len(hits)
            unique.update(hits)
        print '\tfound = ' + str(found)
        print '\tunique = ' + str(len(unique))
        
    def identify_bins(self, moldb, bins):            
        print "Checking discrete identification"
        T = IntervalTree(bins) # store bins in an interval tree
        found = 0
        ambiguous = 0
        unambiguous = 0
        for db_entry in moldb:
            matching_bins = T.search(db_entry.mass)
            found = found + len(matching_bins)
            if len(matching_bins) == 1: # exactly one matching bin
                ambiguous = ambiguous+1
            elif len(matching_bins) > 1: # more than one possible matching bins
                unambiguous = unambiguous+1
        print '\tfound=' + str(found)
        print '\tmatching 1 bin=' + str(unambiguous)    
        print '\tmatching >1 bins=' + str(ambiguous)