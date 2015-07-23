from peak_objects import *
import random
import re
import jsonpickle
import sys


class NetworkSampler(object):
	def __init__(self,peakset):
		self.peakset = peakset
		self.transformations = []
		self.delta = 1.00
		self.load_transformations('all_transformations_masses.txt')
		self.adjacency = {}
		self.create_adjacency(self.transformations)
		self.peakset.posterior_counts = {}
		


	def load_transformations(self,filename):
		self.transformations = []
		with open(filename) as infile:
			for line in infile:
				line = line.rstrip('\r\n')
				splitline = line.split('\t')
				self.transformations.append(Formula(splitline[1]))


	def summarise_posterior(self):
		print
		print "POSTERIOR"
		print
		for m in self.peakset.measurements:
			print "Measurement: " + str(m.id)
			for k in m.annotations:
				print "  " + str(k.formula) + "(" + str(m.annotations[k]) + "): " + str(1.0*self.peakset.posterior_counts[m][k]/self.n_samples)


	def prob_only_sample(self,record = True,verbose = False):
		self.n_samples += 1
		for m in self.peakset.measurements:
			if m.annotations != {}:
				tempprobs = {}
				totalprob = 0.0

				for a in self.adjacency[self.assignment[m]]:
					self.in_degree[a] -= 1

				for k in m.annotations:
					tempprobs[k] = m.annotations[k]
					totalprob+=tempprobs[k]
				u = random.random()*totalprob
				cumprob = 0
				choosepos = -1
				for k in m.annotations:
					cumprob += m.annotations[k]
					choose = k
					if u <= cumprob:
						break

				self.assignment[m] = choose
				for a in self.adjacency[choose]:
					self.in_degree[a] += 1

				if record:
					self.peakset.posterior_counts[m][choose] += 1

				if verbose:
					print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula) + "(" + str(self.peakset.posterior_counts[m][choose]) + ")"

	def multiple_network_sample(self,n_its,record=True,verbose=False):
		for i in range(n_its):
			self.network_sample(record,verbose)

	def network_sample(self,record = True,verbose = False):
		self.n_samples += 1
		for m in self.peakset.measurements:
			if m.annotations != {}:
				tempprobs = {}
				totalprob = 0.0

				for a in self.adjacency[self.assignment[m]]:
					self.in_degree[a] -= 1


				for k in m.annotations:
					tempprobs[k] = m.annotations[k] * (self.delta + self.in_degree[k])
					totalprob+=tempprobs[k]

				u = random.random()*totalprob
				cumprob = 0.0
				choosepos = -1
				for k in m.annotations:
					cumprob += m.annotations[k] * (self.delta + self.in_degree[k])
					choose = k
					if u <= cumprob:
						break

				self.assignment[m] = choose
				for a in self.adjacency[choose]:
					self.in_degree[a] += 1

				if record:
					self.peakset.posterior_counts[m][choose] += 1

				if verbose:
					print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula) + "(" + str(self.peakset.posterior_counts[m][choose]) + ")"


	def initialise_sampler(self,verbose = False):
		self.n_samples = 0
		self.assignment = {}
		self.peakset.posterior_counts = {}
		self.in_degree = {}
		for a in self.peakset.annotations:
			self.in_degree[a] = 0
		for m in self.peakset.measurements:
			if m.annotations != {}:
				self.peakset.posterior_counts[m] = {}
				tempprobs = {}
				totalprob = 0.0
				for k in m.annotations:
					self.peakset.posterior_counts[m][k] = 0
					tempprobs[k] = m.annotations[k]
					totalprob+=tempprobs[k]
				u = random.random()*totalprob
				cumprob = 0.0
				choosepos = -1
				for k in m.annotations:
					cumprob += m.annotations[k]
					choose = k
					if u <= cumprob:
						break

				self.assignment[m] = choose

				for a in self.adjacency[choose]:
					self.in_degree[a] += 1

				if verbose:
					print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula)



	def create_adjacency(self,verbose=False):
		print "Creating adjacency structure. This might take some time..."
		import itertools
		total_found = 0
		for a in self.peakset.annotations:
			self.adjacency[a] = []
		# Loop over annotations
		for a1,a2 in itertools.combinations(self.peakset.annotations,2):
			match_t = self.can_transform(a1,a2)
			if match_t!=None:
				if verbose:
					print "Found match: " + str(a1.formula) + " -> " + str(match_t.formula) + " -> " + str(a2.formula)
				self.adjacency[a1].append(a2)
				self.adjacency[a2].append(a1)
				total_found += 2

		print "Found " + str(total_found) + " (sparsity ratio = " + str(1.0*total_found/(1.0*len(self.peakset.annotations)**2)) + ")"

	def get_all_transforms(self,a1,a2):
		tlist = []
		for t in self.transformations:
			poshit = 1
			neghit = 1
			for a in t.atoms:
				if a1.formula.atoms[a] - a2.formula.atoms[a] != t.atoms[a]:
					poshit = 0
				if a2.formula.atoms[a] - a1.formula.atoms[a] != t.atoms[a]:
					neghit = 0
				if poshit == 0 and neghit == 0:
					break
			if poshit == 1 or neghit == 1:
				tlist.append(t)
		return tlist


	def can_transform(self,a1,a2):
		for t in self.transformations:
			poshit = 1
			neghit = 1
			for a in t.atoms:
				if a1.formula.atoms[a] - a2.formula.atoms[a] != t.atoms[a]:
					poshit = 0
				if a2.formula.atoms[a] - a1.formula.atoms[a] != t.atoms[a]:
					neghit = 0
				if poshit == 0 and neghit == 0:
					break
			if poshit == 1 or neghit == 1:
				return t

		if poshit == 0 and neghit == 0:
			return None

	def dump_output(self,outstream = sys.stdout):
		for m in self.peakset.measurements:
			total_prob = 0.0
			total_count = 0
			if m.annotations == {}:
				continue
			for an in m.annotations:
				total_prob += m.annotations[an]
				total_count += self.peakset.posterior_counts[m][an]
			print >> outstream, "Measurement: {}".format(m.id)
			for an in m.annotations:
				print >> outstream, "\t<<{}>>,<<{}>>,Prior: {:.4f}, Posterior {:.4f}, Degree {}".format(
									an.formula,an.name,m.annotations[an]/total_prob,
									1.0*self.peakset.posterior_counts[m][an]/total_count,self.in_degree[an])


if __name__ == '__main__':
	a = NistOutput('nist_out.txt')
	a.initialise_sampler()
	a.network_sample(100)
	test = jsonpickle.encode(a)
	f = open('picklemodel.txt','w')
	f.write(test)
	# f = open('picklemodel.txt','r')
	# a = jsonpickle.decode(f.read())
	