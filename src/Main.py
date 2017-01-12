from huffman import *

__version__ = "dev"

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
