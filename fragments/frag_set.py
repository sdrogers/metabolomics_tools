import numpy as np
from peak_objects import *

import re
class FragSet(object):
	def __init__(self):
		self.measurements = []
		self.min_relation_size = 2

	def load_from_file(self,filename):
		import pandas as pd
		print "Loading from " + filename
		df = pd.DataFrame.from_csv(filename,sep='\t')
		print "Loaded {} peaks".format(len(df.index))
		rid = np.array(df['relation.id'].tolist())
		print "...with {} relations".format(len(np.unique(rid)))
		mass = np.array(df.index)
		rt = np.array(df['RT'].tolist())
		# Assume the first data column is in position 1 (rt is in 0)
		intensity = np.array(df.iloc[:,1].tolist())

		self.measurements = []

		for r in np.unique(rid):
			peak_pos = np.where(rid == r)[0]
			n_peaks = len(peak_pos)
			if n_peaks >= self.min_relation_size:
				newpeakset = []
				for pos in peak_pos:
					if intensity[pos]>0:
						newpeakset.append(Peak(mass[pos],rt[pos],intensity[pos]))
				if len(newpeakset)>= self.min_relation_size:
					newmeasurement = Measurement(r)
					newmeasurement.add_peak_set(PeakSet(newpeakset))
					self.measurements.append(newmeasurement)

	def make_msp(self,filename,nl = '\r\n'):
		with open(filename,'w') as outfile:
			for m in self.measurements:
				outfile.write("NAME: relation id {}{}".format(m.id,nl))
				outfile.write("DB#: 0{}".format(nl))
				outfile.write("Comments: None{}".format(nl))
				outfile.write("Num Peaks: {}{}".format(m.peakset.n_peaks,nl))
				for peak in m.peakset.peaks:
					outfile.write("{} {}{}".format(peak.mass,peak.norm_intensity,nl))
				outfile.write(nl)

	def query_nist(self):
		# a = system('C:\\2013_06_04_MSPepSearch_x32\\MSPepSearch.exe  M /HITS 3 /PATH C:\\NIST14\\MSSEARCH /MAIN mainlib /INP output.MSP /OUT nist_out.txt /COL pz,cf')
		from subprocess import call
		print "Querying Nist"
		print "\tMaking temporary msp file"
		self.make_msp('temp.MSP')
		st = "C:\\2013_06_04_MSPepSearch_x32\\MSPepSearch.exe  M /HITS 3 /PATH C:\\NIST14\\MSSEARCH /MAIN mainlib /INP temp.MSP /OUT temp_nist.txt /COL pz,cf"
		
		print "\tCalling NIST"
		call(st)
		self.load_annotations('temp_nist.txt')


	def load_annotations(self,filename,correct_gcms_derivatives = False):
		# Load the annotations
		print "\tParsing NIST output"
		# Create a list of IDs for easy access
		mids = [m.id for m in self.measurements]
		current_id = -1
		current_pos = -1

		
		self.annotations = []

		with open(filename,'r') as infile:
			for line in infile:
				m_find = re.search('relation id',line)
				if m_find != None:
					current_id = int(re.findall('relation id (\d*)',line)[0])
					current_pos = [i for i,m in enumerate(mids) if m == current_id]
					current_pos = current_pos[0]

				m_find = re.search('Hit \d*',line)
				if m_find != None:
					name_form = re.findall('<<(.*?)>>',line)
					prob = float(re.findall('Prob: ([0-9]*\.[0-9]*)',line)[0])

					previous_pos = [i for i,l in enumerate(self.annotations) if l.name==name_form[0]]


					if len(previous_pos) == 0:
						# Create the new annotation
						if correct_gcms_derivatives:
							# Check for presence of silicon
							newannotation = Annotation(name_form[1],name_form[0])
							if newannotation.formula.atoms['Si'] == 0:
								# Don't store this as it doesn't have any Silicon
								pass
							else:
								newannotation.formula.correct_gcms_derivatives()
								self.annotations.append(newannotation)
								self.measurements[current_pos].annotations[self.annotations[-1]] = prob
						else:
							self.annotations.append(Annotation(name_form[1],name_form[0]))
							self.measurements[current_pos].annotations[self.annotations[-1]] = prob
					else:
						self.measurements[current_pos].annotations[self.annotations[previous_pos[0]]] = prob

			print "\tLoaded " + str(len(self.annotations)) + " unique annotations"
		
			



if __name__ == '__main__':
	a = FragSet()
	a.load_from_file('mzMATCHoutput.txt')
	a.query_nist()

	