import re

if __name__ == '__main__':
	cas = []
	with open('nist_out.txt') as outfile:
		for line in outfile:
			a = re.search('Hit',line)
			if a != None:
				b = re.findall('CAS:\s*(\d*-\d*-\d*);',line)
				cas.append(b[0])
				

	with open('caslist.txt','w') as outfile:
		for c in cas:
			outfile.write("%s\n" % c)