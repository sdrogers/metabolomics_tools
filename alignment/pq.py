from Queue import PriorityQueue

if __name__ == "__main__":

    q = PriorityQueue()
    q.put((2,"a"))
    q.put((0.6,"b"))
    q.put((0.1,"c"))
    q.put((0.6,"d"))
    q.put((0.19,"e"))
    while not q.empty():
        print q.get()

    q = PriorityQueue()
    q.put((110.0, None))
    q.put((111.0, None))
    q.put((102.0, None))
    q.put((110.0, None))
    q.put((101.0, None))
    q.put((120.0, None))    
    q.put((121.0, None))    
    q.put((123.0, None))    
    q.put((112.0, None))    
    tol = 5
    
    groups = {}
    k = 0
    group = []
    while not q.empty():
        current = q.get()
        group.append(current)
        if len(q.queue) > 0:
            head = q.queue[0]
            if abs(current[0]-head[0])>tol:
                groups[k] = group
                group = []
                k += 1
        else:
            groups[k] = group
            
    for key in groups:
        print "Group %d members %s" % (key, groups[key])