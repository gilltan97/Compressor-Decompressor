class PriorityQueue:
  """
  A container ADT that remove objects according
  to their priority.

  All the items put in this queue should be be tuples of object
  HuffmanNode and int. Tuple with the smallest int value will be
  prioritized and removed. The ties will be dealt in First In
  First Our Order(FIFO).

  This ADT is used to build Tree of HuffmanNode's in function
  huffman_tree
  """
  def __init__(self):
    """(PriorityQueue) -> None
    Initialize a queue with zero items
    """
    self._queue = []

  def enqueue(self, obj):
    """(PriorityQueue, tuple(HuffmanNode, int)) -> None
    Add obj to this PriorityQueue

    >>> pq = PriorityQueue()
    >>> pq.is_empty()
    True
    >>> pq.enqueue((HuffmanNode(2, None, None), 6))
    >>> pq.enqueue((HuffmanNode(3, None, None), 4))
    >>> pq.is_empty()
    False
    """
    self._queue.append(obj)

  def dequeue(self):
    """(PriorityQueue) -> tuple
    Remove and return the object with most priority

    >>> pq = PriorityQueue()
    >>> pq.enqueue((HuffmanNode(2, None, None), 6))
    >>> pq.enqueue((HuffmanNode(3, None, None), 4))
    >>> pq.dequeue()
    (HuffmanNode(3, None, None), 4)
    >>> pq.is_empty()
    False
    >>> pq.dequeue()
    (HuffmanNode(2, None, None), 6)
    >>> pq.is_empty()
    True
    """
    # Add the int from second index of each tuple
    all_freq = []
    for item in self._queue:
      all_freq.append(item[1])

    # Returns the tuple with minimum second index value
    for item in self._queue:
      if item[1] == min(all_freq):
        self._queue.remove(item)
        return item

  def is_empty(self):
    """(PriorityQueue) -> bool
    Return True iff no more objects are left in
    this Priority Queue.

    >>> pq = PriorityQueue()
    >>> pq.is_empty()
    True
    >>> pq.enqueue((HuffmanNode(2, None, None), 6))
    >>> pq.is_empty()
    False
    """
    return len(self._queue) == 0
