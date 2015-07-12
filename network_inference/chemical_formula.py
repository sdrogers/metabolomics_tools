import re


class Formula(object):
	def __init__(self,formula):
		self.atom_names = ['C','H','N','O','P','S','Cl','I','Br','Si','F']
		self.formula = formula
		self.atoms = {}
		for atom in self.atom_names:
			self.atoms[atom] = self.get_atoms(atom)


	def get_atoms(self,atom_name):
		# Do some regex matching to find the numbers of the important atoms
		ex = atom_name + '\d*'
		m = re.search(ex,self.formula)
		if m == None:
			return 0
		else:
			ex = atom_name + "(\d*)"
			m2 = re.findall(ex,m.group(0))[0]
			if len(m2) == 0:
				return 1
			else:
				return int(m2)

	def __repr__(self):
		return self.formula

	def __str__(self):
		return self.formula


if __name__ == '__main__':
	a = Formula("C6H12O6")
	print a.atoms['H']
