import numpy as np

class Peak(object):
	def __init__(self,pid,mass,rt,intensity):
		self.pid = pid
		self.mass = mass
		self.rt = rt
		self.intensity = intensity

class BetaLike(object):
	def __init__(self,alp_in,bet_in,alp_out,bet_out,p0_in,p0_out):
		self.alp_in = alp_in
		self.bet_in = bet_in
		self.alp_out = alp_out
		self.bet_out = bet_out
		self.p0_in = p0_in
		self.p0_out = p0_out
		self.log_p0_in = np.log(p0_in)
		self.log_p0_out = np.log(p0_out)

	def comp_like(self,peak1,peak2,adjacency):
		pass

	def __str__(self):
		return "Beta"


class CorrCluster(object):
	def __init__(self,like_object,peaks,adjacency,alpha=1):
		self.like_object = like_object
		self.peaks = peaks
		self.adjacency = adjacency
		self.alpha = alpha
		self.N = len(self.peaks)
		self.base_like()

	def base_like(self):
		self.base_like = {}

		for p in self.peaks:
			like = self.N*like_object.log_p0_out
			for q in self.adjacency[p]:
				like += np.log(1-self.like_object.p0_out) + (self.alp_out-1.0)*np.log(adjacency[p][q]) + (self.bet_out-1.0)*np.log(1-adjacency[p][q])
				like -= selg.log_p0_out
			self.base_like[p] = like
			

	def __str__(self):
		return "Corr cluster objects with {} peaks and a {} likelihood".format(len(self.peaks),self.like_object)