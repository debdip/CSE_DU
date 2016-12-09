#!/usr/bin/env python

from Crypto.Cipher import AES
import base64
import os

def decr():
	myfile=open('aes_key.hex','r')
	key=myfile.read()
	key=key.decode("hex")
	#print(key)
	myfile1=open('aes_iv.hex','r')
	iv=myfile1.read()
	iv=iv.decode("hex")
	#print(iv)
	myfile2=open('aes_ciphertext.hex','r')
	cipher=myfile2.read()
	cipher=cipher.decode("hex")

	decryption_suite = AES.new(key, AES.MODE_CBC, iv )
	plain_text = decryption_suite.decrypt(cipher)
	text_file = open("Solution02.txt", "w")
	text_file.write(plain_text)
	text_file.close()


decr()
