import numpy as np

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
