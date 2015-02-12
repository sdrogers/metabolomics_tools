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