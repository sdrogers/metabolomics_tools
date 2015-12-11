ELECTRON_MASS = 0.00054857990924

class Transformation(object):
	def __init__(self,name,adduct_mass,charge = 1,multiplicity = 1,fragment_mass = 0,isotope_diff = 0):
		self.name = name
		self.adduct_mass = adduct_mass
		self.charge = charge
		self.multiplicity = multiplicity
		self.fragment_mass = fragment_mass
		self.isotope_diff = isotope_diff

	def transform(self,peak):
		M = peak.mass*self.charge + self.fragment_mass + self.charge*ELECTRON_MASS - self.adduct_mass
		M /= (1.0*self.multiplicity)
		M -= self.isotope_diff
		return M

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.__str__()

def load_from_file(file_name):
	import yaml,re
	transformations = []
	all_vals = yaml.load(open(file_name,'r'))
	# Loop over adducts and fragments
	for tr in all_vals['transformations']:
		multiplicity = all_vals['transformations'][tr]['n']
		charge = 0
		adduct_mass = 0
		for a in all_vals['transformations'][tr]['g']:
			if a in all_vals['charges']:
				charge += all_vals['charges'][a]
			if a in all_vals['masses']:
				adduct_mass += all_vals['masses'][a]
		transformations.append(Transformation(tr,adduct_mass,charge=charge,multiplicity=multiplicity))

		for i in all_vals['isotopes']:
			isotope_diff = all_vals['masses'][i]
			newname = "{} [{}]".format(tr,i)
			transformations.append(Transformation(newname,adduct_mass,charge=charge,multiplicity=multiplicity,isotope_diff=isotope_diff))

		for f in all_vals['fragments']:
			fragment_mass = all_vals['masses'][f]
			splitname = tr.split('+',1)
			newname = "[{}-{}]+{}".format(splitname[0],f,splitname[1])
			transformations.append(Transformation(newname,adduct_mass,charge=charge,multiplicity=multiplicity,fragment_mass=fragment_mass))	
			for i in all_vals['isotopes']:
				isotope_diff = all_vals['masses'][i]
				newname = "{} [{}]".format(newname,i)
				transformations.append(Transformation(newname,adduct_mass,charge=charge,
														multiplicity=multiplicity,
														isotope_diff=isotope_diff,
														fragment_mass = fragment_mass))

	# print transformations
	return transformations


if __name__=='__main__':
	load_from_file('pos_transformations.yml')