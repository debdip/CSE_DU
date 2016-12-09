#!/usr/bin/env python

def WHA(st):
  
	
	Mask=0x3FFFFFFF
	outHash=0	
	for byte in range(len(st)):
		intermediate_value = ((ord(st[byte])^0xCC)<< 24)| ((ord(st[byte])^ 0x33)<< 16)|((ord(st[byte])^ 0xAA)<< 8)|(ord(st[byte])^ 0x55)
		outHash =(outHash & Mask) + (intermediate_value & Mask)
			
	return hex(outHash)	
def WHAT(st):
	print(st)
	print(WHA(st))	
	s=''.join([ st[x:x+2][::-1] for x in range(0, len(st), 2) ]) #alternate string
	print(s)
	print(WHA(s))
WHAT("Hello world!")
WHAT("I am Groot.")

