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

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

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