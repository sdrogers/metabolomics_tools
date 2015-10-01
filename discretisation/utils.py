import re
import numpy as np
from interval_tree import IntervalTree

def print_all_attributes(another_object):
    ''' Prints all the attributes of another_object as [key1=value1, key2=value2, ...] '''
    sb = []
    for key in sorted(another_object.__dict__):
        value = another_object.__dict__[key]
        if isinstance(value, list) or isinstance(value, set):
            sb.append("len({key})={value}".format(key=key, value=len(value)))            
        else:
            sb.append("{key}={value}".format(key=key, value=value))
    return ', '.join(sb)

def natural_sort(l): 
    ''' Sorts list l in a natural order, e.g. 1, 2, 10, 20 ''' 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)    

def num(s):
    if isinstance(s, np.ndarray):
        return np.asscalar(s)    
    try:
        return int(s)
    except ValueError:
        return float(s)

def as_scalar(s):
    if isinstance(s, np.ndarray):
        return np.asscalar(s)    
    else:
        return float(s)
    
def timer(msg, start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(msg + " {:0>2} hours {:0>2} minutes {:05.6f} seconds".format(int(hours),int(minutes),seconds))    

def mass_match(mass, other_masses, tol):
    return np.abs((mass-other_masses)/mass)<tol*1e-6

def rt_match(rt, other_rts, tol):
    return np.abs(rt-other_rts)<tol

def mass_range(mass_centre, mass_tol):
    # must be the same as mass_match()
    interval = mass_centre * mass_tol * 1e-6
    mass_start = mass_centre - interval
    mass_end = mass_centre + interval
    return (mass_start, mass_end)

def mass_centre(mass_start, mass_tol):
    mass_centre = mass_start / (1 - mass_tol * 1e-6)
    return mass_centre    

def rt_range(rt_centre, rt_tol):
    # must be the same as rt_match()
    rt_start = rt_centre - rt_tol
    rt_end = rt_centre + rt_tol
    return (rt_start, rt_end)

# def db_hit(database,mass,tol):
    # returns database hits for the mass at tol ppm
    
def db_hit(database, query_mass):
    ''' Returns database hits for the query_mass. 
        Each DatabaseEntry in database now knows its own range of begin and end masses
        The tolerance ppm for that range is specified when the DatabaseEntry is created.
        See test_discretisation.py and identification.py for example usage.
        
        Args:
         - database: a list of DatabaseEntry objects
         - query_mass: the mass to query

        Returns:
         a list of DatabaseEntry objects {e} where e.get_begin() < query_mass < e.get_end()
    '''
    T = IntervalTree(database)
    hits = T.search(query_mass)
    return hits