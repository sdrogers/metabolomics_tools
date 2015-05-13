from Queue import Queue
from heapq import heappush, heappop

# from http://code.activestate.com/recipes/87369-priority-queue/
class PriorityQueue(Queue):

    # Initialize the queue representation
    def _init(self, maxsize):
        self.size = 0
        self.maxsize = maxsize
        self.queue = []

    # Put a new item in the queue
    def _put(self, item):
        self.size = self.size + 1
        return heappush(self.queue, item)

    # Get an item from the queue
    def _get(self):
        self.size = self.size - 1
        return heappop(self.queue)

if __name__ == "__main__":
    q = PriorityQueue()
    q.put((2,"a"))
    q.put((0.6,"b"))
    q.put((0.1,"c"))
    q.put((0.6,"d"))
    q.put((0.19,"e"))
    while not q.empty():
        print q.get()
