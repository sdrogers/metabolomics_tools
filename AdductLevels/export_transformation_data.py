from discretisation.continuous_mass_clusterer import ContinuousVB
from discretisation.models import HyperPars
from discretisation.preprocessing import FileLoader
from discretisation import utils

def load_and_cluster(input_file, database_file, transformation_file, mass_tol, rt_tol):
        print "Loading " + input_file
        loader = FileLoader()        
        peak_data = loader.load_model_input(input_file, database_file, transformation_file, mass_tol, rt_tol, make_bins=True)
        print "Clustering " + input_file
        hp = HyperPars()
        cluster_model = ContinuousVB(peak_data, hp)
        cluster_model.n_iterations = 20
        print cluster_model
        cluster_model.run()    
        return peak_data, cluster_model
    
def get_feature(peak_data, cluster_model, db_entry, t_find):

    # find cluster membership for all peaks with transformation t_find
    cluster_membership = (cluster_model.Z>0.5)
    trans_mat = peak_data.possible.multiply(cluster_membership)
    trans_mat = (trans_mat==t_find+1)    
    trans_count = (trans_mat).sum(0)

    # match prior bin masses against database entry's mass
    prior_masses = peak_data.prior_masses
    mass_ok = utils.mass_match(db_entry.mass, prior_masses, identify_tol)
    ks = np.flatnonzero(mass_ok)
    
    # enumerate all the possible features
    features = []
    for k in ks:
        if trans_count[0, k]==0:
            continue
        else:
            temp = np.nonzero(trans_mat[:, k])
            peak_idx = temp[0]
            for n in peak_idx:
                f = peak_data.features[n]
                features.append(f)
                
    # if there are multiple features, pick the most intense one
    if len(features)==1:
        return features[0]
    elif len(features)>1: 
        intenses = np.array([f.intensity for f in features])
        most_intense = intenses.argmax()
        return features[most_intense]
    else:
        return None
    
def plot_heatmap(database, dilutions, transformations, filemap, t_find):
    
    print "Finding " + str(transformations[t_find])
    D = len(database)
    L = len(dilutions)    
    results = np.zeros((D, L))   

    # for all db entries
    for d in range(D):
        db_entry = database[d]
        # for all dilution files
        for l in range(L):
            dil = dilutions[l]
            # find the peak
            load_res = filemap[dil]
            peak_data = load_res[0]
            cluster_model = load_res[1]
            f = get_feature(peak_data, cluster_model, db_entry, t_find)
            if f is not None:
                results[d, l] = math.log(f.intensity) 
    # sort the results by first column
    sres = results[results[:, 0].argsort()]    
    # make heatmap
    heatmap = plt.pcolor(sres)   
    plt.show()
    return results

basedir = '.'
dilutions = ["1:1", "1:5", "1:10", "1:50", "1:100", "1:1000"]
input_files = [
    basedir + '/dilutions_data/Positive/std1/Std1_1.csv',
    basedir + '/dilutions_data/Positive/std1/Std1_1in5.csv',
    basedir + '/dilutions_data/Positive/std1/Std1_1in10.csv',
    basedir + '/dilutions_data/Positive/std1/Std1_1in50.csv',    
    basedir + '/dilutions_data/Positive/std1/Std1_1in100.csv',
    basedir + '/dilutions_data/Positive/std1/Std1_1in1000.csv'
]
# dilutions = ["1:1"]
# input_files = [
#     basedir + '/dilutions_data/Positive/std1/Std1_1.csv',
# ]
assert len(input_files) == len(dilutions)
database_file = basedir + '/database/std1_mols.csv'
transformation_file = basedir + '/mulsubs/mulsub2.txt'    
mass_tol = 2
rt_tol = 5

L = len(dilutions)    
filemap = {}
for j in range(L):
    dil = dilutions[j]
    input_file = input_files[j]
    load_res = load_and_cluster(input_file, database_file, transformation_file, mass_tol, rt_tol)
    filemap[dil] = load_res    
    
identify_tol = 2

# get the database and transformations objects from any loaed file
any_entry = filemap[filemap.keys()[0]]
any_peak_data = any_entry[0]
database = any_peak_data.database
transformations = any_peak_data.transformations
for t in range(len(transformations)):
    plot_heatmap(database, dilutions, transformations, filemap, t) 