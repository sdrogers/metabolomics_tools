from models import PeakData
import numpy as np
import scipy.sparse as sp

class ContinuousGibbs:
	def __init__(self,peak_data,hyper_pars):
		self.possible = peak_data.possible
		self.transformed = peak_data.transformed
		self.n_samples = 100
		self.hyper_pars = hyper_pars
		self.rt = peak_data.rt
		self.prior_rt = peak_data.rt
		self.prior_mass = peak_data.precursor_mass
		self.n_peaks = self.prior_mass.size


		# This is peak to cluster assignment
		self.Z = np.arange(self.n_peaks)[:,None]
		self.cluster_size = np.ones_like(self.Z)
		self.cluster_rt_sum = self.rt
		self.cluster_mass_sum = self.prior_mass

		print 'Continuous Clusterer initialised'

	def run(self):
		print "Running the clustering"

		# Find the peaks that we need to re-sample

		todo = np.nonzero((self.possible>0).sum(1)>1)[0]
		
		for samp in np.arange(self.n_samples):
			if samp%10 == 0:
				self.report(samp)

			for i in np.arange(todo.size):
				peak = todo[0,i]
				current_cluster = self.Z[peak]
				self.cluster_size[current_cluster]-=1
				self.cluster_rt_sum[current_cluster]-=self.rt[peak]
				self.cluster_mass_sum[current_cluster]-=self.transformed[peak,current_cluster]

				# Assign to a random cluster

				possible_clusters = np.nonzero(self.possible[peak,:])[1]
				like = np.log((self.hyper_pars.alpha/self.n_peaks) + self.cluster_size[possible_clusters])
				like += self.comp_rt_like(peak,possible_clusters)
				like += self.comp_mass_like(peak,possible_clusters)
				post = np.exp(like - like.max())
				post = post/post.sum()

				# Add random stuff
				pos = np.nonzero(np.random.rand()<post.cumsum())[0][0]
				new_cluster = possible_clusters[pos]

				self.Z[peak] = new_cluster
				self.cluster_size[new_cluster]+=1
				self.cluster_rt_sum[new_cluster]+=self.rt[peak]
				self.cluster_mass_sum[new_cluster]+=self.transformed[peak,new_cluster]

	def comp_mass_like(self,peak,possible_clusters):
		posterior_precision = self.hyper_pars.mass_prior_prec + self.hyper_pars.mass_prec*self.cluster_size[possible_clusters]
		posterior_mean = (1.0/posterior_precision)*(self.hyper_pars.mass_prior_prec*self.prior_mass[possible_clusters] + self.hyper_pars.mass_prec*self.cluster_mass_sum[possible_clusters])
		predictive_precision = 1.0/(1.0/posterior_precision + 1.0/self.hyper_pars.mass_prec)
		return self.log_of_norm_pdf(self.transformed[peak,possible_clusters].toarray().T,posterior_mean,predictive_precision)

	def comp_rt_like(self,peak,possible_clusters):
		posterior_precision = self.hyper_pars.rt_prior_prec + self.hyper_pars.rt_prec*self.cluster_size[possible_clusters]
		posterior_mean = (1.0/posterior_precision)*(self.hyper_pars.rt_prior_prec*self.prior_rt[possible_clusters] + self.hyper_pars.rt_prec*self.cluster_rt_sum[possible_clusters])
		predictive_precision = 1.0/(1.0/posterior_precision + 1.0/self.hyper_pars.rt_prec)
		return self.log_of_norm_pdf(self.rt[peak],posterior_mean,predictive_precision)

	def log_of_norm_pdf(self,x,mu,prec):
		return -0.5*np.log(2*np.pi) + 0.5*np.log(prec) - 0.5*prec*(x-mu)**2

	def report(self,samp):
		print "Sample " + str(samp) + " biggest cluster: " + str(self.cluster_size.max()) + " (" + str(self.cluster_size.argmax()) + ")"

	def set_n_samples(self,n_samples):
		self.n_samples = n_samples

	def __repr__(self):
		return "Gibbs sampler for continuous mass model\n" + self.hyper_pars.__repr__() + \
		"\nn_samples = " + str(self.n_samples)