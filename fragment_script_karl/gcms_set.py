import numpy as np
from peak_objects import *
class GCMSSet(object):
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
		pass
			



if __name__ == '__main__':
	a = GCMSSet()
	a.load_from_file('mzMATCHoutput.txt')
	a.make_msp('gcms.msp',nl = '\r\n')

	