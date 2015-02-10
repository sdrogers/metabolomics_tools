from collections import namedtuple
import csv

DatabaseEntry = namedtuple('DatabaseEntry', ['db_id', 'name', 'formula', 'mass'])
Feature = namedtuple('Feature', ['feature_id', 'mass', 'rt', 'intensity'])
Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul'])

class Feature(object):
        
    def __init__(self, feature_id, mass, rt, intensity):
        self.feature_id = feature_id
        self.mass = mass
        self.rt = rt
        self.intensity = intensity
        self.gt_metabolite = None
        self.gt_adduct = None
        
    def __repr__(self):
        return "Feature id=" + str(self.feature_id) + " mass=" + str(self.mass) + \
            " rt=" + str(self.rt) + " intensity=" + str(self.intensity) + \
            " gt_metabolite=" + str(self.gt_metabolite) + " gt_adduct=" + str(self.gt_adduct)

class FileLoader:
    def load_features(self, input_file):
        """ Load peak features """
        features = []
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=':')
            next(reader, None)  # skip the headers
            for elements in reader:
                feature = Feature(feature_id=self.num(elements[0]), mass=self.num(elements[1]), \
                                  rt=self.num(elements[2]), intensity=self.num(elements[3]))
                features.append(feature)
        return features
    
    def load_features_sima(self, input_file):
        """ Load peak features """
        features = []
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            feature_id = 1
            for elements in reader:
                mass = self.num(elements[0])
                charge = self.num(elements[1])
                intensity = self.num(elements[2])
                rt = self.num(elements[3])
                gt_peak_id = self.num(elements[4])
                gt_metabolite_id = self.num(elements[5])
                gt_adduct_type = elements[6]
                feature = Feature(feature_id, mass, rt, intensity)
                feature.gt_metabolite = gt_metabolite_id
                feature.gt_adduct = gt_adduct_type
                features.append(feature)
                feature_id = feature_id + 1
        return features

    def load_database(self, database):
        moldb = []
        with open(database, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for elements in reader:
                mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                    mass=self.num(elements[3]))
                moldb.append(mol)
        return moldb
    
    def load_transformation(self, transformation):
        transformations = []
        with open(transformation, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 1
            for elements in reader:
                trans = Transformation(trans_id=i, name=elements[0], sub=self.num(elements[1]), \
                                       mul=self.num(elements[2]))
                transformations.append(trans)
                i = i + 1
        return transformations        
    
    def num(self, s):
        try:
            return int(s)
        except ValueError:
            return float(s)

class MassBin:
    def __init__(self, bin_id, start_mass, end_mass, rt):
        self.bin_id = bin_id
        self.start_mass = start_mass
        self.end_mass = end_mass
        self.rt = rt
        self.features = []
        self.molecules = set()
    def get_id(self):
        return self.bin_id
    def get_begin(self):
        return self.start_mass
    def get_end(self):
        return self.end_mass
    def get_rt(self):
        return self.rt
    def add_feature(self, feature):
        self.features.append(feature)
    def remove_feature(self, feature):
        if feature in self.features: 
            self.features.remove(feature)
    def get_features(self):
        return self.features
    def get_features_count(self):
        return len(self.features)
    def get_features_rt(self):
        total_rt = 0
        for feature in self.features:
            total_rt = total_rt + feature.rt
        return total_rt
    def get_molecules(self):
        return self.molecules
    def add_molecule(self, molecule):
        self.molecules.add(molecule)
    def __repr__(self):
        return 'MassBin id=' + str(self.bin_id) + ' mass=(' + str(self.start_mass) + \
            ', ' + str(self.end_mass) + ') num_features=' + str(len(self.features)) + \
            ' num_molecules=' + str(len(self.molecules))

class IntervalTree:
    """ 
    Interval tree implementation
    from http://zurb.com/forrst/posts/Interval_Tree_implementation_in_python-e0K
    """
    def __init__(self, intervals):
        self.top_node = self.divide_intervals(intervals)
 
    def divide_intervals(self, intervals):
 
        if not intervals:
            return None
 
        x_center = self.center(intervals)
 
        s_center = []
        s_left = []
        s_right = []
 
        for k in intervals:
            if k.get_end() < x_center:
                s_left.append(k)
            elif k.get_begin() > x_center:
                s_right.append(k)
            else:
                s_center.append(k)
 
        return Node(x_center, s_center, self.divide_intervals(s_left), self.divide_intervals(s_right))
        
 
    def center(self, intervals):
        fs = sort_by_begin(intervals)
        length = len(fs)
 
        return fs[int(length / 2)].get_begin()
 
    def search(self, begin, end=None):
        if end:
            result = []
 
            for j in xrange(begin, end + 1):
                for k in self.search(j):
                    result.append(k)
                result = list(set(result))
            return sort_by_begin(result)
        else:
            return self._search(self.top_node, begin, [])
    def _search(self, node, point, result):
        
        for k in node.s_center:
            if k.get_begin() <= point <= k.get_end():
                result.append(k)
        if point < node.x_center and node.left_node:
            for k in self._search(node.left_node, point, []):
                result.append(k)
        if point > node.x_center and node.right_node:
            for k in self._search(node.right_node, point, []):
                result.append(k)
 
        return list(set(result))
 
class Interval:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        
    def get_begin(self):
        return self.begin
    def get_end(self):
        return self.end
 
class Node:
    def __init__(self, x_center, s_center, left_node, right_node):
        self.x_center = x_center
        self.s_center = sort_by_begin(s_center)
        self.left_node = left_node
        self.right_node = right_node
 
def sort_by_begin(intervals):
    return sorted(intervals, key=lambda x: x.get_begin())
