class Queue:
  """
  A Container ADT with First In First Out(FIFO) property.

  The ADT is used to visit the tree of HuffmanNode's in
  level order in function improve_tree.
  """
  def __init__(self):
    """(Queue) -> None

    Initialize an empty Queue
    """
    self._queue = []

  def enqueue(self, obj):
    """(Queue, object) -> None
    Add an object to this Queue

    >>> q = Queue()
    >>> q.is_empty()
    True
    >>> q.enqueue(HuffmanNode(5))
    >>> q.enqueue(HuffmanNode(5))
    >>> q.is_empty()
    False
    """
    self._queue.append(obj)

  def dequeue(self):
    """(Queue) -> object
    Remove and return the bottom most object in this Queue

    >>> q = Queue()
    >>> q.is_empty()
    True
    >>> q.enqueue(HuffmanNode(5))
    >>> q.is_empty()
    False
    >>> q.enqueue(HuffmanNode(7))
    >>> q.is_empty()
    False
    >>> q.dequeue()
    HuffmanNode(5, None, None)
    >>> q.is_empty()
    False
    >>> q.dequeue()
    HuffmanNode(7, None, None)
    >>> q.is_empty()
    True
    """
    return self._queue.pop(0)

  def is_empty(self):
    """(Queue) -> bool
    Return True iff no more objects are left in this Queue

    >>> q = Queue()
    >>> q.is_empty()
    True
    >>> q.enqueue(HuffmanNode(5))
    >>> q.enqueue(HuffmanNode(5))
    >>> q.is_empty()
    False
    """
    return len(self._queue) == 0
