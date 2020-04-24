# CompressorDecompressor
[![License: MIT](https://img.shields.io/badge/Python-3-green.svg)]()

A tool to compress/decompress files of any format using the huffman coding.
It compresses the file and generates a compressed file with ```.huf``` extension and decompresses 
that ```.huf``` file back into its original state given the commands to `compress` and `decompress`. 

## Usage
#### Compressing a file
```bash
python3 Main.py
Press c to compress or u to uncompress: c
File to compress: test-files/[file]
```
#### Decompressing a file
```bash 
python3 Main.py
Press c to compress or u to uncompress: u
File to uncompress:  test-files/[file].huf
```
