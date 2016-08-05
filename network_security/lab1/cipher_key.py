#finding the best decription key for the given encripted text manually
#!/usr/bin/env python # ubuntu
from Crypto.Cipher import AES
import base64
import os
myfile2=open('aes_weak_ciphertext.hex','r') #input from hex text
cipher=myfile2.read()
cipher=cipher.decode("hex")  #decode hex 

for j in range(32):
    iv=format(0,"032x").decode("hex")
    key=format(j,"064x").decode("hex") 
    decryption_suite = AES.new(key, AES.MODE_CBC, iv )
    plain_text = decryption_suite.decrypt(cipher)
    print(j)
    print("  Plain text--"+plain_text)
text_file = open("Solution03.hex", "w")
text_file.write(format(30,"064x")) # write the best decription key in hex format
text_file.close()
