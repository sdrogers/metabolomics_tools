import sys

from scipy.special import psi

import numpy as np
import plotting
import scipy.sparse as sp


class ContinuousGibbs:
	# This is the continuous mass gibbs sampler
	def __init__(self,peak_data,hyper_pars):
		self.possible = peak_data.possible
		self.transformed = peak_data.transformed
		self.n_samples = 20
		self.n_burn = 10
		self.hyper_pars = hyper_pars
		self.rt = np.copy(peak_data.rt)
		self.prior_rt = np.copy(peak_data.rt)
		self.prior_mass = np.copy(peak_data.precursor_masses)
		self.n_peaks = self.prior_mass.size


		# This is peak to cluster assignment
		self.Z = sp.lil_matrix((self.n_peaks,self.n_peaks),dtype=np.float)
		self.cluster_size = np.ones(self.n_peaks)[:,None]
		self.cluster_rt_sum = np.copy(self.rt)
		self.cluster_mass_sum = np.copy(self.prior_mass)

		print 'Continuous Clusterer initialised'


	def run(self):
		# I should factorise this a bit more I think...

		print "Running the clustering"

		# Find the peaks that we need to re-sample
		Znk = np.arange(self.n_peaks)[:,None]

		todo = np.nonzero((self.possible>0).sum(1)>1)[0]
		print str(todo.size) + " peaks to be re-sampled"
		for samp in np.arange(self.n_samples):
			if samp%1 == 0:
				plotting.print_cluster_size(self.cluster_size, samp)
				sys.stdout.flush()

			# for i in np.arange(todo.size):
			for i in np.arange(todo.size):
				peak = todo[0,i]
				current_cluster = Znk[peak]
				self.cluster_size[current_cluster]-=1
				self.cluster_rt_sum[current_cluster]-=self.rt[peak]
				self.cluster_mass_sum[current_cluster]-=self.transformed[peak,current_cluster]
				
				# Assign to a random cluster

				# possible_clusters = np.nonzero(self.possible[peak,:])[1]
				possible_clusters = self.possible.getrowview(peak).nonzero()[1]
				like = np.log((self.hyper_pars.alpha/self.n_peaks) + self.cluster_size[possible_clusters])
				like += self._comp_rt_like(peak,possible_clusters)
				like += self._comp_mass_like(peak,possible_clusters)

				post = np.exp(like - like.max())
				post = post/post.sum()

				# Add random stuff
				pos = np.nonzero(np.random.rand()<post.cumsum())[0][0]
				new_cluster = possible_clusters[pos]


				Znk[peak] = new_cluster
				self.cluster_size[new_cluster]+=1
				self.cluster_rt_sum[new_cluster]+=self.rt[peak]
				self.cluster_mass_sum[new_cluster]+=self.transformed[peak,new_cluster]
				if samp>=self.n_burn:
					self.Z[peak,new_cluster] += 1.0

		self.Z /= (self.n_samples-self.n_burn)

		# This sets the probabilities to one for the peaks that were not resampled
		not_done = np.nonzero((self.possible>0).sum(1)==1)[0]
		for i in np.arange(not_done.size):
			peak = not_done[0,i]
			cluster = self.possible.getrowview(peak).nonzero()[1]
			self.Z[peak,cluster] = 1.0

		# we also need a consistent set of cluster precursor masses with precisions
		self.cluster_rt_prec = self.hyper_pars.rt_prior_prec + self.cluster_size*self.hyper_pars.rt_prec
		self.cluster_rt_mean = (1.0/self.cluster_rt_prec)*(self.hyper_pars.rt_prior_prec*self.prior_rt + self.hyper_pars.rt_prec*self.cluster_rt_sum)

		self.cluster_mass_prec = self.hyper_pars.mass_prior_prec + self.cluster_size*self.hyper_pars.mass_prec
		self.cluster_mass_mean = (1.0/self.cluster_mass_prec)*(self.hyper_pars.mass_prior_prec*self.prior_mass + self.hyper_pars.mass_prec*self.cluster_mass_sum)


	def _comp_mass_like(self,peak,possible_clusters):
		posterior_precision = self.hyper_pars.mass_prior_prec + self.hyper_pars.mass_prec*self.cluster_size[possible_clusters]
		posterior_mean = (1.0/posterior_precision)*(self.hyper_pars.mass_prior_prec*self.prior_mass[possible_clusters] + self.hyper_pars.mass_prec*self.cluster_mass_sum[possible_clusters])
		predictive_precision = 1.0/(1.0/posterior_precision + 1.0/self.hyper_pars.mass_prec)
		return self._log_of_norm_pdf(self.transformed[peak,possible_clusters].toarray().T,posterior_mean,predictive_precision)

	def _comp_rt_like(self,peak,possible_clusters):
		posterior_precision = self.hyper_pars.rt_prior_prec + self.hyper_pars.rt_prec*self.cluster_size[possible_clusters]
		posterior_mean = (1.0/posterior_precision)*(self.hyper_pars.rt_prior_prec*self.prior_rt[possible_clusters] + self.hyper_pars.rt_prec*self.cluster_rt_sum[possible_clusters])
		predictive_precision = 1.0/(1.0/posterior_precision + 1.0/self.hyper_pars.rt_prec)
		return self._log_of_norm_pdf(self.rt[peak],posterior_mean,predictive_precision)

	def _log_of_norm_pdf(self,x,mu,prec):
		return -0.5*np.log(2*np.pi) + 0.5*np.log(prec) - 0.5*prec*(x-mu)**2


	def __repr__(self):
		return "Gibbs sampler for continuous mass model\n" + self.hyper_pars.__repr__() + \
		"\nn_samples = " + str(self.n_samples)



class ContinuousVB:
	# This is the variational bayes implementation
	def __init__(self,peak_data,hyper_pars):
		self.possible = peak_data.possible
		self.transformed = peak_data.transformed
		self.n_iterations = 100
		self.hyper_pars = hyper_pars
		self.rt = np.copy(peak_data.rt)
		self.prior_rt = np.copy(peak_data.rt)
		self.prior_mass = peak_data.precursor_masses.copy()
		self.n_peaks = self.prior_mass.size
		self.precursor_rts = peak_data.precursor_rts

		print 'Continuous Clusterer initialised - wow'

	def run(self):
		print "Running the clustering"

		# Find the peaks that we need to re-sample

		todo = np.nonzero((self.possible>0).sum(1)>1)[0]
		print str(todo.size) + " peaks to be re-sampled"

		self.Z = sp.identity(self.n_peaks,format="lil")

		# print "Started inefficient matrix nonsense"

		# precursor_rts = sp.lil_matrix((self.n_peaks,self.n_peaks))
		# for i in np.arange(self.n_peaks):
		# 	pos = np.nonzero(self.possible[i,:])[0]
		# 	precursor_rts[i,pos] = self.rt[pos].T

		# print "Finished inefficient matrix nonsense"


		# These lines dont do anything
		self.precursor_rts.tocsr()
		self.possible.tocsr()
		self.transformed.tocsr()

		for it in np.arange(self.n_iterations):
			if it%1 == 0:
				print "Iteration " + str(it)
				sys.stdout.flush()

			
			sZ = np.array(self.Z.sum(0))
			hyps = sZ + self.hyper_pars.alpha/self.n_peaks
			self.EPi = hyps/hyps.sum()
			self.ELogPi = (psi(hyps) - psi(hyps.sum())).T

			# update RT mean
			b = self.hyper_pars.rt_prior_prec + self.hyper_pars.rt_prec*sZ
			self.hyper_pars.rt_prior_prec*self.prior_rt.T + self.hyper_pars.rt_prec*(self.Z.multiply(self.precursor_rts)).sum(0)
			self.EMuRT = (1.0/b)*(self.hyper_pars.rt_prior_prec*self.prior_rt.T + np.array(self.hyper_pars.rt_prec*(self.Z.multiply(self.precursor_rts)).sum(0)))
			self.EMuRT2 = (1.0/b) + np.square(self.EMuRT)

			# Update mass mean
			d = self.hyper_pars.mass_prior_prec + self.hyper_pars.mass_prec*sZ
			self.EMuMass = (1.0/d)*(self.hyper_pars.mass_prior_prec*self.prior_mass.T + np.array(self.hyper_pars.mass_prec*(self.Z.multiply(self.transformed)).sum(0)))
			self.EMuMass2 = (1.0/d) + np.square(self.EMuMass)

			# for plotting
			self.cluster_rt_mean = self.EMuRT.T
			self.cluster_rt_prec = b.T
			self.cluster_mass_mean = self.EMuMass.T
			self.cluster_mass_prec = d.T

			oldQZ = sp.lil_matrix(self.Z,copy=True)
		

			for i in np.arange(todo.size):
				
				thisRow = todo[0,i]
				thisPos = self.possible.getrowview(thisRow).nonzero()[1]
				
				temp = self.ELogPi[thisPos].T
				
				# thisRT = np.array(self.precursor_rts[thisRow,thisPos].toarray())
				thisTr = np.array(self.transformed.getrowview(thisRow).data[0])[:,None].T
				thisRT = np.tile(self.rt[thisRow],(1,thisTr.size))
				

				temp -= 0.5*self.hyper_pars.rt_prec*(np.square(thisRT) - 2*thisRT*self.EMuRT[0,thisPos] + self.EMuRT2[0,thisPos])
				temp -= 0.5*self.hyper_pars.mass_prec*(np.square(thisTr) - 2*thisTr*self.EMuMass[0,thisPos] + self.EMuMass2[0,thisPos])

				temp = np.exp(temp - temp.max())
				temp = temp/temp.sum()
				self.Z[thisRow,thisPos] = temp
				# print t2a-t2,t2b-t2a,t3-t2b,t4-t3
			QChange = ((oldQZ-self.Z).data**2).sum()
			print "Change in Z: " + str(QChange)




	def __repr__(self):
		return "Variational Bayes for continuous mass model\n" + self.hyper_pars.__repr__() + \
		"\nn_iterations = " + str(self.n_iterations)

