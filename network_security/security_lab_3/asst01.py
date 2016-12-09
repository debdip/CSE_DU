#!/usr/bin/env python

def decrip(text):
	key="DTXCUIQVMRWFZGPLHAOYKNEBSJ"
	dic={'D':'A','T':'B','X':'C','C':'D','U':'E','I':'F','Q':'G','V':'H','M':'I','R':'J','W':'K','F':'L','Z':'M','G':'N','P':'O','L':'P','H':'Q','A':'R','O':'S','Y':'T','K':'U','N':'V','E':'W','B':'X','S':'Y','J':'Z'}
	plain=''
	for i in range(len(text)):
		if(text[i] in dic):
			plain+=dic[text[i]]
		else: plain+=text[i]	
	print(plain)

decrip("LFDXMCP CPZMGQP RPOU XDAAUADO  YVMO MYDFMDG XDFFUC YVUSAU CPMGQ D IPKA YUGPAO YPKA  SPKAU GKZTUA IPKA")
