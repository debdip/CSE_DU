#!/usr/bin/env python

from Crypto.Hash import SHA256
h = SHA256.new()

h.update('LE PENNED THE LINES THEY LOOKED NERVOUS AS CATS NERVOUS AS A COUPLE OF CATS ON A HOT TIN ROOF')
text1=h.hexdigest()
h1=SHA256.new()
h1.update('HE PENNED THE LINES THEY LOOKED NERVOUS AS CATS NERVOUS AS A COUPLE OF CATS ON A HOT TIN ROOF')
text2=h1.hexdigest()
int1=int(text1,16)
int2=int(text2,16)
int3=int1^int2
diff= bin(int3).count("1")
print hex(diff)

