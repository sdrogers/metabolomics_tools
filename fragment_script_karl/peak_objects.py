import numpy as np
import re

class Peak(object):
	def __init__(self,mass,rt,intensity):
		self.mass = mass
		self.rt = rt
		self.intensity = intensity
		self.norm_intensity = intensity

	def __repr__(self):
		return "Mass: {}, RT: {}, Intensity: {}".format(self.mass,self.rt,self.intensity)

class PeakSet(object):
	def __init__(self,peaks):
		self.peaks = peaks
		masses = [p.mass for p in peaks]
		rts = [p.rt for p in peaks]
		intensities = [p.intensity for p in peaks]
		basepos = np.argmax(np.array(intensities))
		self.basepeak = peaks[basepos]

		self.normalise_intensities()
		self.n_peaks = len(peaks)


	def normalise_intensities(self):
		for p in self.peaks:
			p.norm_intensity = 100.0*p.intensity/self.basepeak.intensity


class Measurement(object):
	def __init__(self,thisid):
		self.id = thisid
		self.annotations = {}

	def add_peak_set(self,peakset):
		self.peakset = peakset



class Annotation(object):
	def __init__(self,formula,name):
		self.formula = Formula(formula)
		self.name = name

class Formula(object):
	def __init__(self,formula):
		self.atom_names = ['C','H','N','O','P','S','Cl','I','Br','Si','F','D']
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


