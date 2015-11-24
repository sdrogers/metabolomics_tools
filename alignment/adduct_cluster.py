import numpy as np
import sys
from discretisation.mulsubs import transformation
import pylab as plt

class Peak(object):

	def __init__(self,mass,rt,intensity):
		self.mass = mass
		self.rt = rt
		self.intensity = intensity

class Cluster(object):
	
	def __init__(self,mHPeak,M,id,mass_tol = 5, rt_tol = 10):
		self.mHPeak = mHPeak
		self.N = 1
		self.mass_sum = M
		self.rt_sum = self.mHPeak.rt
		self.M = M
		self.prior_rt_mean = mHPeak.rt
		self.prior_mass_mean = M

		delta = mass_tol*M/1e6
		var = (delta/3.0)**2
		self.prior_mass_precision = 1.0/var
		self.mass_precision = 1.0/var

		delta = rt_tol
		var = (delta/3.0)**2
		self.prior_rt_precision = 1.0/var
		self.rt_precision = 1.0/var

		self.id = id

	def compute_rt_like(self,rt):
		post_prec = self.prior_rt_precision + self.N*self.rt_precision
		post_mean = (1.0/post_prec)*(self.prior_rt_precision*self.prior_rt_mean + self.rt_precision*self.rt_sum)
		pred_prec = (1.0/(1.0/post_prec + 1.0/self.rt_precision))
		self.mu_mass = post_mean
		return -0.5*np.log(2*np.pi) + 0.5*np.log(pred_prec) - 0.5*pred_prec*(rt-post_mean)**2

	def compute_mass_like(self,mass):
		post_prec = self.prior_mass_precision + self.N*self.mass_precision
		post_mean = (1.0/post_prec)*(self.prior_mass_precision*self.prior_mass_mean + self.mass_precision*self.mass_sum)
		pred_prec = (1.0/(1.0/post_prec + 1.0/self.mass_precision))
		self.mu_rt = post_mean
		return -0.5*np.log(2*np.pi) + 0.5*np.log(pred_prec) - 0.5*pred_prec*(mass-post_mean)**2


class Possible(object):

	def __init__(self,cluster,transformation,transformed_mass,rt):
		self.count = 0
		self.cluster = cluster
		self.transformation = transformation
		self.transformed_mass = transformed_mass
		self.rt = rt

class Transformation(object):

	def __init__(self,name,mul,sub,de):
		self.name = name
		self.mul = mul
		self.sub = sub
		self.de = de

	def transform(self,peak):
		return (peak.mass - self.sub)/self.mul + self.de

class AdductCluster(object):

	def __init__(self,rt_tol = 5,mass_tol = 1,transformation_file = 'mulsubs/pos_transformations.yml',
				alpha = 1,verbose = 0, mh_biggest = True):
		self.mass_tol = mass_tol
		self.rt_tol = rt_tol
		self.transformation_file = transformation_file
		self.load_transformations()
		self.alpha = alpha
		self.verbose = verbose
		self.nSamples = 0
		self.mh_biggest = mh_biggest

	def load_transformations(self):
		self.transformations = transformation.load_from_file(self.transformation_file)
		self.MH = None
		for t in self.transformations:
			if t.name=="M+H":
				self.MH = t

	def init_from_file(self,filename):
		peak_list = []
		with open(filename,'r') as f:
			heads = f.readline()
			for line in f:
				line = line.split('\t')
				mass = float(line[0]);
				rt = float(line[1]);
				intensity = float(line[2]);
				peak_list.append(Peak(mass,rt,intensity))

		print "Loaded {} peaks from {}".format(len(peak_list),filename);
		self.init_from_list(peak_list)
		
	def init_from_list(self,peak_list):

		self.peaks = []
		self.clusters = []
		self.possible = {}
		self.Z = {}
		self.todo = []
		self.clus_poss = {}
		current_id = 0
		for p in peak_list:
			self.peaks.append(p)
			c = Cluster(p,self.MH.transform(p),current_id,
				mass_tol = self.mass_tol,rt_tol = self.rt_tol)
			current_id += 1
			self.clusters.append(c)
			poss = Possible(c,self.MH,self.MH.transform(p),p.rt)
			self.possible[p] = {}
			self.possible[p] = [poss]
			self.Z[p] = poss
			self.clus_poss[c] = [poss]

		print "Created {} clusters".format(len(self.clusters))
		self.K = len(self.clusters)

		if self.mh_biggest:

			for n in range(len(peak_list)):
	
				p = peak_list[n]
				if n%500==0:
					print "Assigning possible transformations %d/%d" % (n, len(peak_list))
					sys.stdout.flush()
					
				for c in self.clusters:
					if p is c.mHPeak:
						continue
					if p.intensity > c.mHPeak.intensity:
						continue
					else:
						t = self.check(p,c)
						if not t == None:
							poss = Possible(c,t,t.transform(p),p.rt)
							self.possible[p].append(poss)
							self.clus_poss[c].append(poss)

		else:
			
			# first stage is to check the MH biggest constraint
			transformed_into = {}
			for n in range(len(peak_list)):
	
				p = peak_list[n]
				if n%500==0:
					print "Assigning first-stage possible transformations %d/%d" % (n, len(peak_list))
					sys.stdout.flush()
					
				for c in self.clusters:
					if p is c.mHPeak:
						continue
					if p.intensity > c.mHPeak.intensity:
						continue
					else:
						t = self.check(p,c)
						if t is not None and t.name != 'M+H':
							poss = Possible(c,t,t.transform(p),p.rt)
							self.possible[p].append(poss)
							self.clus_poss[c].append(poss)
							if c in transformed_into:
								transformed_into[c].append(p._get_key())
							else:
								transformed_into[c] = [p._get_key()]
	
			# second stage is without the M+H constraint
			for n in range(len(peak_list)):
	
				p = peak_list[n]
				if n%500==0:
					print "Assigning second-stage possible transformations %d/%d" % (n, len(peak_list))
					sys.stdout.flush()
					
				for c in self.clusters:
					
					if p is c.mHPeak:
						continue

					if c in transformed_into and p._get_key() in transformed_into[c]:
						continue # prevent duplicates

					allow = False
					if c in transformed_into and p._get_key() not in transformed_into[c]:
						existing = transformed_into[c]
						if len(existing)>0:
							# only if there's another existing transformation that obeys the stricter criteria
							allow = True
					elif p.intensity < c.mHPeak.intensity*2:
						# or if the peak intensity is at most twice as large as the M+H peak intensity
						allow = True

					t = self.check(p,c)
					if allow and t is not None and t.name != 'M+H':
						poss = Possible(c,t,t.transform(p),p.rt)
						self.possible[p].append(poss)
						self.clus_poss[c].append(poss)

		for p in peak_list:
			if len(self.possible[p])>1:
				self.todo.append(p)

		print "{} peaks to be re-sampled in stage 1".format(len(self.todo))

	def reset_counts(self):

		for p in self.peaks:
			for poss in self.possible[p]:
				poss.count = 0
		self.nSamples = 0

	def init_vb(self):
		
		for cluster in self.clusters:
			cluster.sumZ = 0.0
			cluster.pi = 1.0+self.alpha/(1.0*self.K)
			cluster.pi /= (self.K + self.alpha)
		for peak in self.peaks:
			for poss in self.possible[peak]:
				if poss.transformation == self.MH:
					self.Z[peak] = poss
					poss.prob = 1.0
					poss.cluster.sumZ += 1.0
				else:
					poss.prob = 0.0

	def vb_step(self):
		
		# Update means
		for cluster in self.clusters:
			cluster.sumZ = 0.0
			cluster.sum_rt = 0.0
			cluster.sum_mass = 0.0
			for poss in self.clus_poss[cluster]:
				poss.cluster.sumZ += poss.prob
				poss.cluster.sum_rt += poss.prob*poss.rt
				poss.cluster.sum_mass += poss.prob*poss.transformed_mass

			prec = cluster.prior_rt_precision + cluster.sumZ*cluster.rt_precision
			cluster.mu_rt = (1.0/prec)*(cluster.prior_rt_precision*cluster.prior_rt_mean + cluster.rt_precision*cluster.sum_rt)
			cluster.mu_rt_2 = (1.0/prec) + cluster.mu_rt**2

			prec = cluster.prior_mass_precision + cluster.sumZ*cluster.mass_precision
			cluster.mu_mass = (1.0/prec)*(cluster.prior_mass_precision*cluster.prior_mass_mean + cluster.mass_precision*cluster.sum_mass)
			cluster.mu_mass_2 = (1.0/prec) + cluster.mu_mass**2

			cluster.pi = (cluster.sumZ + self.alpha/(1.0*self.K))/(self.K + self.alpha)

		# Update Z
		for peak in self.todo:
			max_prob = -1e6
			for poss in self.possible[peak]:
				poss.prob = np.log(poss.cluster.pi)
				poss.prob += -0.5*np.log(2*np.pi) + 0.5*np.log(poss.cluster.rt_precision) - 0.5*poss.cluster.rt_precision*(poss.rt**2 - 2*poss.rt*poss.cluster.mu_rt + poss.cluster.mu_rt_2)
				poss.prob += -0.5*np.log(2*np.pi) + 0.5*np.log(poss.cluster.mass_precision) - 0.5*poss.cluster.mass_precision*(poss.transformed_mass**2 - 2*poss.transformed_mass*poss.cluster.mu_mass + poss.cluster.mu_mass_2)
				if poss.prob > max_prob:
					max_prob = poss.prob

			total_prob = 0.0
			for poss in self.possible[peak]:
				poss.prob = np.exp(poss.prob - max_prob)
				total_prob += poss.prob

			for poss in self.possible[peak]:
				poss.prob /= total_prob

	def multi_sample(self, S):

		for s in range(S):
			self.do_gibbs_sample()

		# Fix the counts for things that don't get re-sampled
		for p in self.peaks:
			if p not in self.todo:
				self.possible[p][0].count += S
				self.possible[p][0].cluster.mu_mass = p.mass
				self.possible[p][0].cluster.mu_rt = p.rt

	def do_gibbs_sample(self):
		
		for p in self.todo:
			
			# Remove from current cluster
			old_poss = self.Z[p]
			old_poss.cluster.N -= 1
			old_poss.cluster.rt_sum -= p.rt
			old_poss.cluster.mass_sum -= old_poss.transformed_mass
			
			post_max = -1e6
			post = []
			for poss in self.possible[p]:
				new_post = poss.cluster.compute_rt_like(p.rt)
				new_post += poss.cluster.compute_mass_like(poss.transformation.transform(p))
				new_post += np.log(poss.cluster.N + (1.0*self.alpha)/(1.0*self.K))
				post.append(new_post)
				if new_post > post_max:
					post_max = new_post

			post = np.array(post)
			post = np.exp(post - post_max)
			post /= post.sum()
			post = post.cumsum()
			pos = np.where(np.random.rand()<post)[0][0]
			
			new_poss = self.possible[p][pos]

			self.Z[p] = new_poss
			new_poss.cluster.N += 1
			new_poss.cluster.rt_sum += p.rt
			new_poss.cluster.mass_sum += new_poss.transformed_mass
			new_poss.count += 1
			
		self.nSamples += 1

	def compute_posterior_probs(self):
		for p in self.peaks:
			for poss in self.possible[p]:
				poss.prob = (1.0*poss.count)/(1.0*self.nSamples) 

	def display_probs(self):
		# Only displays the ones in todo
		for p in self.todo:
			print "Peak: {},{}".format(p.mass,p.rt)
			for poss in self.possible[p]:
				print "\t Cluster {}: {} ({} = {}), prob = {}".format(poss.cluster.id,poss.cluster.M,
					poss.transformation.name,poss.transformed_mass,poss.prob)			

	def check(self,peak,cluster):
		# Check RT first
		if np.abs(peak.rt - cluster.mHPeak.rt) > self.rt_tol:
			return None
		else:
			for t in self.transformations:
				tm = t.transform(peak)
				if np.abs((tm - cluster.M)/cluster.M)*1e6 < self.mass_tol:
					return t
			return None

	def map_assign(self):
		# Assigns all peaks to their most likely cluster
		# This is a bit odd for the VB inference
		for c in self.clusters:
			c.N = 0
			c.rt_sum = 0
			c.mass_sum = 0
		for p in self.peaks:
			possible_clusters = self.possible[p]
			# self.Z[p].cluster.N -= 1
			# self.Z[p].cluster.rt_sum -= p.rt
			# self.Z[p].cluster.mass_sum -= self.Z[p].transformed_mass
			if len(possible_clusters) == 1:
				self.Z[p] = possible_clusters[0]
			else:
				max_prob = 0.0
				for poss in possible_clusters:
					if poss.prob > max_prob:
						self.Z[p] = poss
						max_prob = poss.prob
			self.Z[p].cluster.N += 1
			self.Z[p].cluster.rt_sum += p.rt
			self.Z[p].cluster.mass_sum += self.Z[p].transformed_mass

	def cluster_plot(self,cluster):
		# Find the members and possible objects
		members = []
		possibles = []
		trans = []
		for p in self.peaks:
			if self.Z[p].cluster is cluster:
				members.append(p)
				possibles.append(self.Z[p])
				trans.append(self.Z[p].transformation)

		print "CLUSTER {}".format(cluster.id)
		max_intensity = 0
		for p in range(len(members)):
			print "Peak: {},{},{} -> {},{} (p={})".format(members[p].mass,members[p].rt,members[p].intensity,
												possibles[p].transformation.name,
												possibles[p].transformed_mass,
												possibles[p].prob)
			if members[p].intensity > max_intensity:
				max_intensity = members[p].intensity

		plt.figure()
		for p_ind in range(len(members)):
			p = members[p_ind]
			po = possibles[p_ind]
			plt.plot((p.mass,p.mass),(0,p.intensity/max_intensity),'k-')
			plt.annotate(po.transformation.name,(p.mass,p.intensity/max_intensity),
						(p.mass,0.1+p.intensity/max_intensity),
						arrowprops=dict(arrowstyle='->'),
						textcoords='data')
		
		trans_names = [i.name for i in trans]
		all_trans_names = [i.name for i in self.transformations]
		for this_name in all_trans_names:
			r = (this_name,this_name + " [C13]")
			if r[0] in trans_names and r[1] in trans_names:
				pos_0 = trans_names.index(r[0])
				pos_1 = trans_names.index(r[1])
				print "{}/{} = {}".format(r[0],r[1],members[pos_0].intensity/members[pos_1].intensity)

if __name__=="__main__":
	infile = sys.argv[1]
	ac = AdductCluster(alpha = 1)
	ac.init_from_file(infile)
	ac.multi_sample(1000)
	print ac.nSamples
	ac.compute_posterior_probs()
	ac.display_probs()