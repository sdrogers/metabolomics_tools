from chemical_formula import Formula
import random
import re
class NistOutput(object):
	def __init__(self,filename):
		self.filename = filename
		self.measurements = []
		self.annotations = []
		self.load_output()
		transformations = []
		self.delta = 1.00
		transformations = self.load_transformations('all_transformations_masses.txt')
		# transformations.append(Formula('C5H4N5'))
		# transformations.append(Formula('C2H2O'))
		# transformations.append(Formula('H2O'))
		# transformations.append(Formula('C5H7NO3'))
		# transformations.append(Formula('C12H5'))
		self.adjacency = {}
		self.create_adjacency(transformations)

		# self.initialise_sampler()
		# for i in range(1000):
		# 	self.network_sample()

		# self.summarise_posterior()


	def load_transformations(self,filename):
		transformations = []
		with open(filename) as infile:
			for line in infile:
				line = line.rstrip('\r\n')
				splitline = line.split('\t')
				transformations.append(Formula(splitline[1]))
		return transformations


	def summarise_posterior(self):
		print
		print "POSTERIOR"
		print
		for m in self.measurements:
			print "Measurement: " + str(m.id)
			for k in m.annotations:
				print "  " + str(k.formula) + "(" + str(m.annotations[k]) + "): " + str(1.0*self.posterior_counts[m][k]/self.n_samples)


	def prob_only_sample(self,record = True,verbose = False):
		self.n_samples += 1
		for m in self.measurements:
			tempprobs = {}
			totalprob = 0

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
				self.posterior_counts[m][choose] += 1

			if verbose:
				print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula) + "(" + str(self.posterior_counts[m][choose]) + ")"

	def network_sample(self,record = True,verbose = False):
		self.n_samples += 1
		for m in self.measurements:
			tempprobs = {}
			totalprob = 0

			for a in self.adjacency[self.assignment[m]]:
				self.in_degree[a] -= 1

			for k in m.annotations:
				tempprobs[k] = m.annotations[k] * (self.delta + self.in_degree[k])
				totalprob+=tempprobs[k]
			u = random.random()*totalprob
			cumprob = 0
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
				self.posterior_counts[m][choose] += 1

			if verbose:
				print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula) + "(" + str(self.posterior_counts[m][choose]) + ")"


	def initialise_sampler(self,verbose = False):
		self.n_samples = 0
		self.assignment = {}
		self.posterior_counts = {}
		self.in_degree = {}
		for a in self.annotations:
			self.in_degree[a] = 0
		for m in self.measurements:
			self.posterior_counts[m] = {}
			tempprobs = {}
			totalprob = 0
			for k in m.annotations:
				self.posterior_counts[m][k] = 0
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

			if verbose:
				print "Measurement: " + str(m.id) + " assigned to " + str(choose.formula)


	def create_adjacency(self,transformations):
		import itertools
		total_found = 0
		for a in self.annotations:
			self.adjacency[a] = []
		# Loop over annotations
		for a1,a2 in itertools.combinations(self.annotations,2):
			match_t = self.can_transform(a1,a2,transformations)
			if match_t!=None:
				print "Found match: " + str(a1.formula) + " -> " + str(match_t.formula) + " -> " + str(a2.formula)
				self.adjacency[a1].append(a2)
				self.adjacency[a2].append(a1)
				total_found += 2

		print "Found " + str(total_found) + " (sparsity ratio = " + str(1.0*total_found/(1.0*len(self.annotations)**2)) + ")"


	def can_transform(self,a1,a2,transformations):
		for t in transformations:
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

	def load_output(self):
		print "loading"
		newmeasurement = None
		with open(self.filename) as file:
			for line in file:
				headline = re.search('relation id \d*',line)
				if headline != None:
					print headline.group(0)
					newid = re.findall('relation id (\d*)',headline.group(0))[0]
					newmeasurement = Measurement(newid)					
					self.measurements.append(newmeasurement)
				else:
					# Check for a hit
					hitline = re.search('Hit \d*',line)
					if hitline != None:
						name_form = re.findall('<<(.*?)>>',line)
						prob = float(re.findall('Prob: ([0-9]*\.[0-9]*)',line)[0])
						# Check for this molecule before
						previous_pos = [i for i,l in enumerate(self.annotations) if l.name==name_form[0]]
						if len(previous_pos) == 0:
							self.annotations.append(Annotation(name_form[1],name_form[0],prob))
							newmeasurement.annotations[self.annotations[-1]] = prob
						else:
							newmeasurement.annotations[self.annotations[previous_pos[0]]] = prob
		print "Loaded " + str(len(self.measurements)) + " measurements and " + str(len(self.annotations)) + " unique annotations"

class Measurement(object):
	def __init__(self,thisid):
		self.id = thisid
		self.annotations = {}



class Annotation(object):
	def __init__(self,formula,name,probability):
		self.formula = Formula(formula)
		self.name = name
		self.probability = probability

if __name__ == '__main__':
	a = NistOutput('nist_out.txt')