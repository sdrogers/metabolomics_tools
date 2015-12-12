import numpy as np
import pylab as plt
from scipy.stats import beta
class Peak(object):
	def __init__(self,pid,mass,rt,intensity):
		self.pid = pid
		self.mass = mass
		self.rt = rt
		self.intensity = intensity

	def __str__(self):
		return "({}, {})".format(self.mass,self.rt)

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

	def in_like(self,c):
		l = np.log(1-self.log_p0_in)
		l += (self.alp_in - 1)*np.log(c)
		l += (self.bet_in - 1)*np.log(1-c)
		return l


	def out_like(self,c):
		l = np.log(1-self.log_p0_out)
		l += (self.alp_out - 1)*np.log(c)
		l += (self.bet_out - 1)*np.log(1-c)
		return l

	def plot_like(self):
		plt.figure()
		xv = np.arange(0,1,0.01)
		yin = beta.pdf(xv,self.alp_in,self.bet_in)
		yout = beta.pdf(xv,self.alp_out,self.bet_out)
		plt.plot(xv,yin)
		plt.plot(xv,yout)

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
		self.init_clusterer()

	class Cluster(object):
		def __init__(self):
			self.members = []
			self.size = 0
			self.rt_sum = 0.0

	def init_clusterer(self):
		self.Z = {}
		self.clusters = []
		cl = self.Cluster()
		self.clusters.append(cl)
		for p in self.peaks:
			self.Z[p] = cl
			cl.members.append(p)
			cl.size += 1
			cl.rt_sum += p.rt

		print "Initialised with {} clusters".format(len(self.clusters))

	def base_like(self):
		self.base_like = {}

		for p in self.peaks:
			like = self.N*self.like_object.log_p0_out
			for q in self.adjacency[p]:
				like -= self.like_object.log_p0_out
				like += self.like_object.out_like(self.adjacency[p][q])
			self.base_like[p] = like
			
	def resample_peak_membership(self,peak):
		current_cluster = self.Z[peak]
		current_cluster.members.remove(peak)
		current_cluster.size -= 1
		current_cluster.rt_sum -= peak.rt
		if current_cluster.size == 0:
			self.clusters.remove(current_cluster)


		# allpr = [np.log(i.size)+self.base_like[peak] + i.size*(self.like_object.log_p0_in-self.like_object.log_p0_out) for i in self.clusters]
		# possible_clusters = set([self.Z[n] for n in self.adjacency[peak]])

		# for cluster in possible_clusters:
		# 	pr = np.log(cluster.size) + self.base_like[peak]
		# 	for p in cluster.members:
		# 		if p in self.adjacency[peak]:
		# 			pr += self.like_object.in_like(self.adjacency[peak][p])
		# 			pr -= self.like_object.out_like(self.adjacency[peak][p])
		# 		else:
		# 			pr += self.like_object.log_p0_in
		# 			pr -= self.like_object.log_p0_out
		# 	allpr[self.clusters.index(cluster)] = pr

		# allpr.append(np.log(self.alpha) + self.base_like[peak])
		# probs = np.array(allpr)
		# ma = probs.max()
		# probs = np.exp(probs - ma)
		# probs = np.divide(probs,probs.sum()).cumsum()


		probs = []
		max_prob = -1e6
		for i,cluster in enumerate(self.clusters):
			pr = np.log(cluster.size) + self.base_like[peak]
			for p in cluster.members:
				if p in self.adjacency[peak]:
					pr += self.like_object.in_like(self.adjacency[peak][p])
					pr -= self.like_object.out_like(self.adjacency[peak][p])
				else:
					pr += self.like_object.log_p0_in
					pr -= self.like_object.log_p0_out
			if pr >= max_prob:
				max_prob = pr
			probs.append(pr)

		probs.append(np.log(self.alpha) + self.base_like[peak])
		if probs[-1] >= max_prob:
			max_prob = probs[-1]



		probs = np.array(probs)
		probs = np.exp(probs - max_prob)
		probs = np.divide(probs,probs.sum()).cumsum()

		new_index = np.where(np.random.rand()<probs)[0][0]

		if new_index < len(self.clusters):
			new_cluster = self.clusters[new_index]
			self.Z[peak] = new_cluster
			new_cluster.members.append(peak)
			new_cluster.size += 1
			new_cluster.rt_sum += peak.rt
		else:
			new_cluster = self.Cluster()
			self.clusters.append(new_cluster)
			new_cluster.members.append(peak)
			new_cluster.size += 1
			self.Z[peak] = new_cluster
			new_cluster.rt_sum = peak.rt
		
	def check(self):
		# Check that the numbers tally!
		print "{} clusters".format(len(self.clusters))
		n = 0
		for cluster in self.clusters:
			n += cluster.size
		if not n == self.N:
			print "ERROR"


	def gibbs_cycle(self):
		for i,peak in enumerate(self.peaks):
			# if i%100 == 0:
			# 	print "\t {}".format(i)
			self.resample_peak_membership(peak)
		# self.resample_peak_membership(self.peaks[0])
		self.check()


	def multi_gibbs_cycle(self,S):
		for s in range(S):
			print "Sample {}".format(s)
			self.gibbs_cycle()

	def get_peaks_by_cluster(self):
		self.clusters = sorted(self.clusters,key=lambda x: x.size,reverse=True)
		ordered_peaks = []
		order = []
		for cluster in self.clusters:
			ordered_peaks += cluster.members
			for p in cluster.members:
				order.append(self.peaks.index(p))
		return ordered_peaks,order


	def greedy(self,thresh=0.75):
		# do a greedy clustering
		self.greedy_clusters = []
		sorted_peaks = sorted(self.peaks,key=lambda x: x.intensity,reverse=True)
		finished = False
		while not finished:
			if len(sorted_peaks) == 0:
				finished = True
				break
			new_cluster = self.Cluster()
			p = sorted_peaks[0]
			new_cluster.members.append(p)
			new_cluster.size = 1
			sorted_peaks.remove(p)
			for q in self.adjacency[p]:
				if q in sorted_peaks:
					if self.adjacency[p][q]>=thresh:
						sorted_peaks.remove(q)
						new_cluster.members.append(q)
						new_cluster.size += 1
			self.greedy_clusters.append(new_cluster)

		self.greedy_clusters = sorted(self.greedy_clusters,key=lambda x: x.size,reverse=True)
		ordered_peaks = []
		order = []
		for cluster in self.greedy_clusters:
			ordered_peaks += cluster.members
			for p in cluster.members:
				order.append(self.peaks.index(p))
		return ordered_peaks,order


	def __str__(self):
		return "Corr cluster objects with {} peaks and a {} likelihood".format(len(self.peaks),self.like_object)