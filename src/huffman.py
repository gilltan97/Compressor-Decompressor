'''
File Name:huffman.py
Author: Tanveer Gill
'''


import doctest, time

from nodes import HuffmanNode, ReadNode


# =======================================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
  """ (int, int) -> int
  Return bit number bit_num from right in byte.

  >>> get_bit(0b00000101, 2)
  1
  >>> get_bit(0b00000101, 1)
  0
  """
  return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
  """ (int) -> str
  Return the representation of a byte as a string of bits.

  >>> byte_to_bits(14)
  '00001110'
  """
  return "".join([str(get_bit(byte, bit_num)) for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
  """ (str) -> int
  Return int represented by bits, padded on right.

  >>> bits_to_byte("00000101")
  5
  >>> bits_to_byte("101") == 0b10100000
  True
  """
  return sum([int(bits[pos]) * (1 << (7 - pos))
              for pos in range(len(bits))])


# ===============================
# Abstract Data type(ADT) classes


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


# =========================
# Functions for compression


def make_freq_dict(text):
  """ (bytes) -> dict of {int: int}
  Return a dictionary that maps each byte in text to its frequency.

  >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
  >>> d == {65: 1, 66: 2, 67: 1}
  True
  """
  frequencies = {}
  for byte in text:
    # If byte already in the frequencies dict
    if byte in frequencies:
      frequencies[byte] += 1

    # If it's bytes first occurrence
    else:
      frequencies[byte] = 1
  return frequencies


def huffman_tree(freq_dict):
  """ (dict of {int: int}) -> HuffmanNode
  Return the root HuffmanNode of a Huffman tree corresponding
  to frequency dictionary freq_dict.

  >>> freq = {2: 6, 3: 4}
  >>> t = huffman_tree(freq)
  >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
  >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
  >>> t == result1 or t == result2
  True
  """
  # Make tuple with pair of Huffman node with each key and frequency of that key
  priority_queue = PriorityQueue()
  for data in freq_dict:
    priority_queue.enqueue((HuffmanNode(data), freq_dict[data]))

  # Merge two nodes from tuples with minimum value at index 2, and-
  # append it back into priority_queue as a tuple with added frequencies-
  # of two node keys on the second index of the tuple.
  while not priority_queue.is_empty():
    node1 = priority_queue.dequeue()
    if priority_queue.is_empty():
      return node1[0]
    node2 = priority_queue.dequeue()
    parent_node = HuffmanNode(None, node1[0], node2[0])
    priority_queue.enqueue((parent_node, node1[1] + node2[1]))


def get_codes(tree):
  """ (HuffmanNode) -> dict of {int: str}
  Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

  >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
  >>> d = get_codes(tree)
  >>> d == {3: "0", 2: "1"}
  True
  """
  def dict_of_codes(tree_, str_=""):
    """
    (HuffmanNode, str) -> dict of {int: str}
    Return dict of encodes mapped to the symbols in a Tree
    """
    # If the tree has 0 nodes
    if tree_ is None:
      return

    # If a node with symbol is found store the symbol
    if tree_.symbol is not None:
      code_dict[tree_.symbol] = str_

    # If node is a leaf node return nothing
    if tree_.is_leaf():
      return

    # If the tree has both of its children
    else:
      dict_of_codes(tree_.left, str_ + "0")
      dict_of_codes(tree_.right, str_ + "1")
  code_dict = {}
  dict_of_codes(tree)
  return code_dict


def number_nodes(tree, numb=0):
  """ (HuffmanNode) -> NoneType
  Number internal nodes in tree according to postorder traversal;
  start numbering at 0.

  >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
  >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
  >>> tree = HuffmanNode(None, left, right)
  >>> number_nodes(tree)
  >>> tree.left.number
  0
  >>> tree.right.number
  1
  >>> tree.number
  2
  """
  # If the tree has zero nodes
  if tree is None:
    return

  # If the tree has 0 internal nodes
  elif tree.is_leaf():
    return

  else:
    # Visit the left subtree and place numb if any internal node exist
    number_nodes(tree.left, numb)
    # Visit the right subtree and place numb if no external node was -
    # found in left subtree. Otherwise, place number in left subtree-
    # external node plus 1.
    number_nodes(tree.right,
                  tree.left.number + 1 if tree.left.number is not None
                  else numb)
    # Place one of the three possible value for the root node from :
    # 1 - numb (if no external node is found in left and right subtree)
    if tree.left.number is None and tree.right.number is None:
      tree.number = numb

    # 2 - value of external node in left subtree + 1 (no external node in right)
    elif tree.right.number is None:
      tree.number = tree.left.number + 1

    # 3 - value of external node right subtree + 1 (no external node in left)
    else:
      tree.number = tree.right.number + 1
  return


def avg_length(tree, freq_dict):
  """ (HuffmanNode, dict of {int : int}) -> float
  Return the number of bits per symbol required to compress text
  made of the symbols and frequencies in freq_dict, using the Huffman tree.

  >>> freq = {3: 2, 2: 7, 9: 1}
  >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
  >>> right = HuffmanNode(9)
  >>> tree = HuffmanNode(None, left, right)
  >>> avg_length(tree, freq)
  1.9
  """
  # Dictionary of binary codes mapped to symbols
  all_codes = get_codes(tree)
  total_freq = sum(list(freq_dict.values()))

  # Sum of all the bits
  total_bits = sum([freq_dict[symbol] * len(all_codes[symbol])
                    for symbol in freq_dict])

  # Average number of bits per symbol
  return total_bits / total_freq


def generate_compressed(text, codes):
  """ (bytes, dict of {int: str}) -> bytes
  Return compressed form of text, using mapping in codes for each symbol.

  >>> d = {0: "0", 1: "10", 2: "11"}
  >>> text = bytes([1, 2, 1, 0])
  >>> result = generate_compressed(text, d)
  >>> [byte_to_bits(byte) for byte in result]
  ['10111000']
  >>> text = bytes([1, 2, 1, 0, 2])
  >>> result = generate_compressed(text, d)
  >>> [byte_to_bits(byte) for byte in result]
  ['10111001', '10000000']
  >>> text = bytes([1,1,1,2,0,0,2,0,2])
  >>> result = generate_compressed(text, d)
  >>> [byte_to_bits(byte) for byte in result]
  ['10101011', '00110110']
  """
  # Make string of all the bits of symbols
  all_bits = ''
  for symbol in text:
    all_bits += codes[symbol]

  # Add 0's to make length of all_bits a multiple of 8 -
  # so that n number of bytes can be created from it
  while len(all_bits) % 8:
    all_bits += '0'

  # Make and return bytes from the string all_bits
  all_bytes = []
  index = 8
  for _ in range(len(all_bits) // 8):
    all_bytes.append(bits_to_byte(all_bits[index - 8: index]))
    index += 8
  return bytes(all_bytes)


def tree_to_bytes(tree):
  """(HuffmanNode) -> bytes
  Return a bytes representation of the Huffman tree rooted at tree.
  The representation should be based on the post-order traversal of tree.
  Precondition: tree has its nodes numbered.

  >>> left = HuffmanNode(3, None, None)
  >>> right = HuffmanNode(2, None, None)
  >>> tree = HuffmanNode(None, left , right)
  >>> number_nodes(tree)
  >>> list(tree_to_bytes(tree))
  [0, 3, 0, 2]
  >>> left_left = HuffmanNode(3, None, None)
  >>> left_right = HuffmanNode(2, None, None)
  >>> left = HuffmanNode(None, left_left , left_right)
  >>> right = HuffmanNode(5)
  >>> tree = HuffmanNode(None, left, right)
  >>> number_nodes(tree)
  >>> list(tree_to_bytes(tree))
  [0, 3, 0, 2, 1, 0, 0, 5]
  >>> left = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
  >>> right_left = HuffmanNode(None, HuffmanNode(0), HuffmanNode(3))
  >>> right_right = HuffmanNode(4)
  >>> right = HuffmanNode(None, right_left, right_right)
  >>> tree = HuffmanNode(None, left, right)
  >>> number_nodes(tree)
  >>> list(tree_to_bytes(tree))
  [0, 1, 0, 2, 0, 0, 0, 3, 1, 1, 0, 4, 1, 0, 1, 2]
  """
  # If the tree has zero nodes
  if tree is None:
    return bytes([])

  # If the tree consists of only one node
  if tree.is_leaf():
    return bytes([])

  # If both the children of the roor node are leaf
  if tree.left.is_leaf() and tree.right.is_leaf():
      return bytes([0, tree.left.symbol,
                    0, tree.right.symbol])

  # If left child of the root node is a leaf
  if tree.left.is_leaf() and not tree.right.is_leaf():
      return (tree_to_bytes(tree.right) +
              bytes([0, tree.left.symbol] +
                    [1, tree.right.number]))

  # If right child of the root is  a leaf
  if tree.right.is_leaf() and not tree.left.is_leaf():
      return (tree_to_bytes(tree.left) +
              bytes([1, tree.left.number] +
                    [0, tree.right.symbol]))

  # If both the children of root node are not leaf
  else:
    return (tree_to_bytes(tree.left) +
            tree_to_bytes(tree.right) +
            bytes([1, tree.left.number] +
                  [1, tree.right.number]))


def num_nodes_to_bytes(tree):
  """ (HuffmanNode) -> bytes
  Return number of nodes required to represent tree, the root of a
  numbered Huffman tree.
  """
  return bytes([tree.number + 1])


def size_to_bytes(size):
  """ (int) -> bytes
  Return the size as a bytes object.

  >>> list(size_to_bytes(300))
  [44, 1, 0, 0]
  """
  # little-endian representation of 32-bit (4-byte)
  # int size
  return size.to_bytes(4, "little")


def compress(in_file, out_file):
  """ (str, str) -> NoneType
  Compress contents of in_file and store results in out_file.
  """
  text = open(in_file, "rb").read()
  freq = make_freq_dict(text)
  tree = huffman_tree(freq)
  codes = get_codes(tree)
  number_nodes(tree)
  print("Bits per symbol:", avg_length(tree, freq))
  result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
            size_to_bytes(len(text)))
  result += generate_compressed(text, codes)
  open(out_file, "wb").write(result)


# ===========================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
  """ (list of ReadNode, int) -> HuffmanNode
  Return the root of the Huffman tree corresponding to node_lst[root_index].
  The function assumes nothing about the order of the nodes in the list.

  >>> lst = [ReadNode(0, 5, 0, 7)]
  >>> generate_tree_general(lst, 0)
  HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None))
  >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 0, 0, 3), \
             ReadNode(1, 1, 0, 4), ReadNode(1, 0, 1, 2)]
  >>> generate_tree_general(lst, 3)
  HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None)), HuffmanNode(None, HuffmanNode(None, HuffmanNode(0, None, None), HuffmanNode(3, None, None)), HuffmanNode(4, None, None)))
  >>> lst = [ReadNode(1,1,0,7), ReadNode(0,2,0,1)]
  >>> generate_tree_general(lst, 0)
  HuffmanNode(None, HuffmanNode(None, HuffmanNode(2, None, None), HuffmanNode(1, None, None)), HuffmanNode(7, None, None))
  >>> lst = [ReadNode(0,2,0,1), ReadNode(1,0,1,2), ReadNode(0,10,0,20)]
  >>> generate_tree_general(lst, 1)
  HuffmanNode(None, HuffmanNode(None, HuffmanNode(2, None, None), HuffmanNode(1, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(20, None, None)))
  """
  # Got to the root node
  root_data = node_lst[root_index]
  root = HuffmanNode()

  # If left child of root is a leaf add it right away
  if not root_data.l_type:
    root.left = HuffmanNode(root_data.l_data)

  # if left child of a root is an internal node go to it's children
  if root_data.l_type:
    root.left = generate_tree_general(node_lst, root_data.l_data)

  # if right child of root is a left add it right away
  if not root_data.r_type:
    root.right = HuffmanNode(root_data.r_data)

  # if left child pf a root is an internal node go to it's children
  if root_data.r_type:
    root.right = generate_tree_general(node_lst, root_data.r_data)
  return root


def generate_tree_postorder(node_lst, root_index):
  """ (list of ReadNode, int) -> HuffmanNode
  Return the root of the Huffman tree corresponding to node_lst[root_index].
  The function assumes that the list represents a tree in postorder.

  >>> lst = [ReadNode(0, 5, 0, 7)]
  >>> generate_tree_postorder(lst, 0)
  HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None))

  >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 0, 0, 3), \
             ReadNode(1, 1, 0, 4), ReadNode(1, 0, 1, 2)]
  >>> generate_tree_postorder(lst, 3)
  HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None)), HuffmanNode(None, HuffmanNode(None, HuffmanNode(0, None, None), HuffmanNode(3, None, None)), HuffmanNode(4, None, None)))
  """
  # Make a new list of read nodes till root index
  new_list = [node_lst[i] for i in range(root_index + 1)]
  stack = Stack()

  while new_list:
    node_data = new_list.pop(0)
    # If the left and right node are leafs merge them and add them-
    # back to the stack
    if not node_data.l_type and not node_data.r_type:
      left = HuffmanNode(node_data.l_data)
      right = HuffmanNode(node_data.r_data)
      stack.add(HuffmanNode(None, left, right))

    # If left node is a leaf, merge it with the top most object in-
    # the stack and put it back to the stack
    elif not node_data.l_type:
      left = HuffmanNode(node_data.l_data)
      right = stack.pop()
      stack.add(HuffmanNode(None, left, right))

    # If right node is a leaf, merge it with the top most object in-
    # the stack and put it back to the stack
    elif not node_data.r_type:
      right = HuffmanNode(node_data.r_data)
      left = stack.pop()
      stack.add(HuffmanNode(None, left, right))

    # If both right and left nodes are internal nodes, merge last two-
    # objects in the stack and put the merged object back into the stack
    elif node_data.r_type and node_data.l_type:
      right = stack.pop()
      left = stack.pop()
      stack.add(HuffmanNode(None, left, right))

  # Merge elements in the stack until only one element is left in the stack.
  while not stack.is_empty():
    node1 = stack.pop()
    if stack.is_empty():
      return node1
    node2 = stack.pop()
    stack.add(HuffmanNode(None, node2, node1))


def generate_uncompressed(tree, text, size):
  """ (HuffmanNode, bytes, int) -> bytes
  Use Huffman tree to decompress size bytes from text.
  """
  all_bytes = []
  all_bits = ""
  curr_node = tree
  index = 0

  # Make string of all the bits
  for byte in text:
    all_bits += byte_to_bits(byte)

  while len(all_bytes) != size:
    # If leaf is found in the tree add it's symbol to all_bytes
    if not curr_node.left and not curr_node.right:
      all_bytes.append(curr_node.symbol)
      curr_node = tree

    # If first bit of of all bits is 1, the go to the right node in the tree
    elif all_bits[index] is '1':
      curr_node = curr_node.right
      index += 1

    # If first bit of of all bits is 0,the go to the left node in the tree
    elif all_bits[index] is '0':
      curr_node = curr_node.left
      index += 1
  return bytes(all_bytes)


def bytes_to_nodes(buf):
  """ (bytes) -> list of ReadNode
  Return a list of ReadNodes corresponding to the bytes in buf.

  >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
  [ReadNode(0, 1, 0, 2)]
  """
  lst = []
  for i in range(0, len(buf), 4):
    l_type = buf[i]
    l_data = buf[i + 1]
    r_type = buf[i + 2]
    r_data = buf[i + 3]
    lst.append(ReadNode(l_type, l_data, r_type, r_data))
  return lst


def bytes_to_size(buf):
  """ (bytes) -> int
  Return the size corresponding to the given 4-byte
  little-endian representation.

  >>> bytes_to_size(bytes([44, 1, 0, 0]))
  300
  """
  return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
  """ (str, str) -> NoneType
  Uncompress contents of in_file and store results in out_file.
  """
  f = open(in_file, "rb")
  num_nodes = f.read(1)[0]
  buf = f.read(num_nodes * 4)
  node_lst = bytes_to_nodes(buf)
  # use generate_tree_general or generate_tree_postorder here
  tree = generate_tree_general(node_lst, num_nodes - 1)
  size = bytes_to_size(f.read(4))
  g = open(out_file, "wb")
  text = f.read()
  g.write(generate_uncompressed(tree, text, size))


# ===============
# Other functions

def improve_tree(tree, freq_dict):
  """(HuffmanNode, dict of {int : int}) -> NoneType
  Improve the tree as much as possible, without changing its shape,
  by swapping nodes. The improvements are with respect to freq_dict.

  >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(99, None, None), \
  HuffmanNode(100, None, None)), \
  HuffmanNode(None, HuffmanNode(101, None, None),\
  HuffmanNode(None, HuffmanNode(97, None, None), HuffmanNode(98, None, None))))
  >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
  >>> improve_tree(tree, freq)
  >>> avg_length(tree, freq)
  2.31
  """
  # Make HuffmanNode's with symbols as keys of freq_dict and put nodes-
  # with most frequency at top and nodes with least frequency at bottom-
  # of a stack
  priority_queue = PriorityQueue()
  for symbol in freq_dict:
    priority_queue.enqueue((HuffmanNode(symbol), freq_dict[symbol]))
  stack = Stack()
  while not priority_queue.is_empty():
    stack.add(priority_queue.dequeue()[0])

  # Traverse the tree in level-order
  queue = Queue()
  queue.enqueue(tree)
  while not queue.is_empty():
    node = queue.dequeue()
    # If leaf is found, swap it's symbol with the top most node in the stack
    if not node.left and not node.right:
      efficient_node = stack.pop()
      node.symbol = efficient_node.symbol
    # Continue to traverse in level-order otherwise
    else:
      queue.enqueue(node.left)
      queue.enqueue(node.right)

if __name__ == "__main__":
  doctest.testmod()

  mode = input("Press c to compress or u to uncompress: ")
  if mode == "c":
    fname = input("File to compress: ")
    start = time.time()
    compress(fname, fname + ".huf")
    print("compressed {} in {} seconds.".format(fname, time.time() - start))
  elif mode == "u":
    fname = input("File to uncompress: ")
    start = time.time()
    uncompress(fname, fname + ".orig")
    print("uncompressed {} in {} seconds.".format(fname, time.time() - start))
