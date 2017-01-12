class Stack:
  """
  A Container ADT with Last In First Out(LIFO) property.

  This ADT is used to construct a tree of HuffmanNode's form
  ReadNodes in function generate_tree_postorder.
  """
  def __init__(self):
    """(Stack) -> None
    Initialize an empty stack
    """
    self._stack = []

  def add(self, obj):
    """(Stack, object) -> None
    Add obj to this Stack

    >>> s = Stack()
    >>> s.is_empty()
    True
    >>> s.add(HuffmanNode(5))
    >>> s.is_empty()
    False
    >>> s.add(HuffmanNode(7))
    >>> s.is_empty()
    False
    >>> s.pop()
    HuffmanNode(7, None, None)
    >>> s.is_empty()
    False
    >>> s.pop()
    HuffmanNode(5, None, None)
    >>> s.is_empty()
    True
    """
    self._stack.append(obj)

  def pop(self):
    """(Stack) -> object
    Remove and return the top most object in this stack
    Add obj to this Stack

    >>> s = Stack()
    >>> s.is_empty()
    True
    >>> s.add(HuffmanNode(5))
    >>> s.add(HuffmanNode(7))
    >>> s.pop()
    HuffmanNode(7, None, None)
    >>> s.is_empty()
    False
    >>> s.pop()
    HuffmanNode(5, None, None)
    >>> s.is_empty()
    True
    """
    return self._stack.pop()

  def is_empty(self):
    """(Stack) -> object
    Return True iff no more objects are left in this stack

    >>> s = Stack()
    >>> s.is_empty()
    True
    >>> s.add(HuffmanNode(5))
    >>> s.is_empty()
    False
    """
    return len(self._stack) == 0
