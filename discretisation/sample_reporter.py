from collections import Counter

def print_cluster_size(cluster_size, samp):
    print "Sample " + str(samp) + " biggest cluster: " + str(cluster_size.max()) + " (" + str(cluster_size.argmax()) + ")"

def print_cluster_sizes(bins, samp, time_taken):
    cluster_sizes = [str(mb.get_features_count()) for mb in bins]
    c = Counter()
    c.update(cluster_sizes) # count frequencies of each size 
    print('SAMPLE %d %4.2fs\t%s' % ((samp+1), time_taken, str(c)))
    
def print_last_sample(bins, feature_annotation):
    """ Print bins (clusters) in descending order of size and the features inside

        Args: 
         - bins: the list of all the PrecursorBin objects (having features inside)
         - feature_annotation: the map of feature -> adduct type annotation

        Returns:
         - None
    """
    
    print 'Last sample report'
    count_empty_bins = 0
    bin_mols = []
    bin_mols_unique = set()
    
    # sort the bin by no. of features first (biggest first)
    bins.sort(key=lambda x: x.get_features_count(), reverse=True)
    # loop through bins and print stuff
    for mass_bin in bins:
        # count molecules annotated to bin
        bin_mols.extend(mass_bin.molecules)
        bin_mols_unique.update(mass_bin.molecules)
        if mass_bin.get_features_count() > 0:
            # print header
            print mass_bin
            for mol in mass_bin.molecules:
                print "\t" + str(mol)
            table = []
            table.append(['feature_id', 'mass', 'rt', 'intensity', 'annotation', 'gt_metabolite', 'gt_adduct'])
            # print features in this mass bin
            for f in mass_bin.features:
                table.append([str(f.feature_id), str(f.mass), str(f.rt), str(f.intensity), 
                              feature_annotation[f], str(f.gt_metabolite), str(f.gt_adduct)])
            __print_table(table)
            print
        else:
            count_empty_bins = count_empty_bins + 1

    print 'Empty bins=' + str(count_empty_bins)
    print 'Occupied bins=' + str(len(bins) - count_empty_bins) 

def __print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print "| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |"