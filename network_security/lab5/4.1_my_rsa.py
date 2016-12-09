#!/usr/bin/env python
from os import chmod
import math
import random
def isPrime(num):
	if num<2:
		return false
	for i in range(2,int(math.sqrt(num))+1):
		if num%i==0:
			return False
	return True
def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi
    
    while e > 0:
        temp1 = temp_phi/e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2
        
        x = x2- temp1* x1
        y = d - temp1 * y1
        
        x2 = x1
        x1 = x
        d = y1
        y1 = y
    
    if temp_phi == 1:
		return d + phi
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
i=0
p=0
q=0 
while(p==0):
	r=random.randint(3,1000)
	if(isPrime(r)==True):
		p=r
while(q==0):
	r=random.randint(3,1000)
	if(isPrime(r)==True):
		q=r
n=p*q
phi=(p-1)*(q-1)
e = random.randrange(1, phi)
g = gcd(e, phi)
while g != 1:
	e = random.randrange(1, phi)
	g = gcd(e, phi)
d = multiplicative_inverse(e, phi)
num=int(raw_input())
encrypt=pow(num,e)%n
decrypt=pow(encrypt,d)%n
print(decrypt)
print(p)
print(q)
print(phi)
print(e)

