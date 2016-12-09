#!/usr/bin/env python 
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA

private_key=RSA.importKey(open("id_rsa","r").read())
public_key=RSA.importKey(open("id_rsa.pub","r").read())

print private_key
print public_key

msg="hello world"

encrypt=public_key.encrypt(msg,32)

print encrypt

decrypt=private_key.decrypt(encrypt)

print decrypt
