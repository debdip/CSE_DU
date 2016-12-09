#!/usr/bin/env python
from Crypto.Cipher import AES
import base64
import os
myfile2=open('aes_weak_ciphertext.hex','r')
cipher=myfile2.read()
cipher=cipher.decode("hex")

for j in range(32):
    iv=format(0,"032x").decode("hex")
    key=format(j,"064x").decode("hex")
    decryption_suite = AES.new(key, AES.MODE_CBC, iv )
    plain_text = decryption_suite.decrypt(cipher)
    print(j)
    print("  Plain text--"+plain_text)
text_file = open("Solution03.hex", "w")
text_file.write(format(30,"064x"))
text_file.close()
