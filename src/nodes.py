'''
File Name: nodes.py
Author: Tanveer Gill
'''

class HuffmanNode:

  """ A node in a Huffman tree.
  Symbols occur only at leaves.
  Each node has a number attribute that can be used for node-numbering.
  """

  def __init__(self, symbol=None, left=None, right=None):
    """(HuffmanNode, int|None, HuffmanNode|None, HuffmanNode|None,
    HuffmanNode|None)

    Create a new HuffmanNode with the given parameters.
    """
    self.symbol = symbol
    self.left, self.right = left, right
    self.number = None

  def __eq__(self, other):
    """(HuffmanNode, HuffmanNode) -> bool

    Return True iff self is equivalent to other.

    >>> a = HuffmanNode(4)
    >>> b = HuffmanNode(4)
    >>> a == b
    True
    >>> b = HuffmanNode(5)
    >>> a == b
    False
    """
    return (type(self) == type(other) and self.symbol == other.symbol and
    self.left == other.left and self.right == other.right)

  def __lt__(self, other):
    """ (HuffmanNode, HuffmanNode) -> bool

    Return True iff self is less than other.
    """
    return False # arbitrarily say that one node is never less than another

  def __repr__(self):
    """(HuffmanNode) -> str

    Return constructor-style string representation.

    """
    return 'HuffmanNode({}, {}, {})'.format(self.symbol,
    self.left, self.right)


  def is_leaf(self):
    """(HuffmanNode) -> bool

    Return True iff self is a leaf.

    >>> t = HuffmanNode(None)
    >>> t.is_leaf()
    True
    """
    return not self.left and not self.right


class ReadNode:

  """
  A node as read from a compressed file.
  Each node consists of type and data information as described in the handout.
  This class offers a clean way to collect this information together for each
  node.
  """

  def __init__(self, l_type, l_data, r_type, r_data):
    """(ReadNode, int, int, int, int)

    Create a new ReadNode with the given parameters.
    """
    self.l_type, self.l_data = l_type, l_data
    self.r_type, self.r_data = r_type, r_data

  def __repr__(self):
    """(ReadNode) -> str

    Return constructor-style string representation.

    """
    return 'ReadNode({}, {}, {}, {})'.format(
    self.l_type, self.l_data, self.r_type, self.r_data)

if __name__ == '__main__':
  import doctest
  doctest.testmod()
