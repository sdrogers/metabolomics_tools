
import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

import com.csvreader.CsvWriter;

// libraries
import domsax.*;
import cmdline.*;

// peakml
import peakml.*;
import peakml.math.*;
import peakml.chemistry.*;

import peakml.io.*;
import peakml.io.peakml.*;
import peakml.io.chemistry.*;

// mzmatch
import mzmatch.ipeak.sort.IdentifyPeaksets;
import mzmatch.util.*;

import mzmatch.ipeak.sort.Data;

public class Group {
	private String inFile = "/Users/simon/Dropbox/BioResearch/Meta_clustering/JoeDataProcessing/Standards/std1 pos/std1-file1.peakml"; 
	private ParseResult result;
	private Data data;
	private Vector<Transformation> transformations;
	private static final Double mass_tol = 5.0;
	private static final Double rt_tol = 4.0;
	private int proton_pos;
	private ArrayList<ArrayList<Possible>> possibles;
	private ArrayList<Integer> needUpdates;
	private Integer[] clusterSizes;
	private Double[] clusterMassSums; 
	private Double[] clusterRTSums;
	private Double[] clusterPriorMeanRT;
	private Double[] clusterPriorMeanMass;
	private Double clusterPriorPrecisionRT = 1.0;
	private Double clusterPriorPrecisionMass = 1000.0;
	private Double precisionRT = 1.0;
	private Double precisionMass = 1000.0;
	private Double alpha = 1.0;
	private Integer[] z;
	private Integer[] tr;
	private Double[] transformed_mass;
	private Random r;
	public Group() {
		try {
			result = PeakMLParser.parse(new FileInputStream(inFile),true);
			data = new Data(result.header,(IPeakSet<IPeak>) result.measurement);
			// Unlog masses 
			for(int i=0;i<data.numPeaksets;i++) {
				for(int j=0;j<data.numReplicates;j++) {
					data.masses[j][i] = Math.exp(data.masses[j][i]);
				}
			}
			System.out.println("Loaded " + data.numPeaksets + " peaksets");
		}catch(Exception e) {
			System.out.println(e);
		}
		loadTransformations();
		proton_pos = -1;
		for(int i=0;i<transformations.size();i++) {
			if(transformations.get(i).getName().equals("M+H")) {
				proton_pos = i;
			}
		}
		System.out.println("Proton pos: " + proton_pos);
		findPossible();

		needUpdates = new ArrayList<Integer>();
		for(int i=0;i<possibles.size();i++) {
			if(possibles.get(i).size()>1) {
				needUpdates.add(i);
			}
		}
		System.out.println("" + needUpdates.size() + " peaks require clustering");

		initialiseClustering();

		r = new Random();
		doSample(100);
		checkStatus();
		summarise();
	}

	private void doSample(int n_samples) {
		for(int i=0;i<n_samples;i++) {
			System.out.println("Sample " + i);
			doSample();
		}
	}
	private void checkStatus() {
		// Firstly check clusterSizes
		int nPeaks = data.numPeaksets;
		int totalPeaks = 0;
		double totalRT = 0.0;
		double totalRTcluster = 0.0;
		double totalMass = 0.0;
		double totalMasscluster = 0.0;
		for(int i=0;i<nPeaks;i++) {
			totalPeaks += clusterSizes[i];
			totalRT += data.retentionTimes[0][i];
			totalRTcluster += clusterRTSums[i];
			totalMasscluster += clusterMassSums[i];
			totalMass += transformations.get(tr[i]).transformMass(data.masses[0][i]);
		}
		assert (totalPeaks==nPeaks): "Bad peak counts: " + totalPeaks + " neq " + nPeaks;
		assert (totalRT==totalRTcluster): "Bad RT sums";
		assert (Math.abs(totalMass-totalMasscluster)<1e-6): "Bad Mass sums: " + totalMass + " neq " + totalMasscluster;
	}
	private void doSample() {
		// Loop over the peaks, re-sampling
		Integer nClusters = data.numPeaksets;
		for(Integer peak: needUpdates) {
			Integer current_cluster = z[peak];
			clusterRTSums[current_cluster] -= data.retentionTimes[0][peak];
			clusterSizes[current_cluster]--;
			clusterMassSums[current_cluster] -= transformed_mass[peak];
			Integer current_transform = tr[peak];
			ArrayList<Possible> p = possibles.get(peak);
			Double[] posterior = new Double[p.size()];
			Double max_posterior = 0.0;
			for(int i=0;i<p.size();i++) {
				int this_cluster = p.get(i).cluster;
				Double prior = Math.log(clusterSizes[this_cluster] + alpha/(1.0*nClusters));
				// Retention time likelihood
				Double precision = clusterPriorPrecisionRT + precisionRT * clusterSizes[this_cluster];
				Double mean = (1.0/precision)*(clusterPriorMeanRT[this_cluster]*clusterPriorPrecisionRT + precisionRT*clusterRTSums[this_cluster]);
				Double pred_precision = 1.0/(1.0/precision + 1.0/precisionRT);
				Double like = -0.5*Math.log(2.0*Math.PI) + 0.5*Math.log(pred_precision);
				like -= 0.5*pred_precision*Math.pow(mean - data.retentionTimes[0][peak],2.0);

				// Mass likelihood
				precision = clusterPriorPrecisionMass + precisionMass * clusterSizes[this_cluster];
				mean = (1.0/precision)*(clusterPriorMeanMass[this_cluster]*clusterPriorPrecisionMass + precisionMass*clusterMassSums[this_cluster]);
				pred_precision = 1.0/(1.0/precision + 1.0/precisionMass);
				like -= 0.5*Math.log(2.0*Math.PI) + 0.5*Math.log(pred_precision);
				like -= 0.5*pred_precision*Math.pow(mean - p.get(i).mass,2.0);

				posterior[i] = prior+like;
				if(i==0 | posterior[i]>max_posterior) {
					max_posterior = posterior[i];
				}
			}
			Double total_posterior = 0.0;
			for(int i=0;i<p.size();i++) {
				posterior[i]-=max_posterior;
				posterior[i] = Math.exp(posterior[i]);
				total_posterior += posterior[i];
			}

			// Normalise and sample from the posterior
			Double u = r.nextDouble();
			Double cumPost = posterior[0]/total_posterior;
			int new_cluster = 0;
			while(u > cumPost) {
				new_cluster++;
				cumPost += posterior[new_cluster]/total_posterior;
			}

			z[peak] = p.get(new_cluster).cluster;
			tr[peak] = p.get(new_cluster).transformation;

			transformed_mass[peak] = p.get(new_cluster).mass;
			clusterSizes[z[peak]] ++;
			clusterRTSums[z[peak]] += data.retentionTimes[0][peak];
			clusterMassSums[z[peak]] += transformed_mass[peak];

		}
	}
	private void summarise() {
		// Displaying clusters with >5 members
		ArrayList<Integer> big = new ArrayList<Integer>();
		ArrayList<Integer> empty = new ArrayList<Integer>();
		HashMap<Integer,Integer> histogram = new HashMap<Integer,Integer>();
		Integer nClusters = data.numPeaksets;
		Integer biggest = 0;
		int biggest_pos = -1;
		for(int i=0;i<nClusters;i++) {
			if(clusterSizes[i]>=5) {
				big.add(i);
			}
			if(clusterSizes[i]==0) {
				empty.add(i);
			}
			Integer count = histogram.get((Integer)clusterSizes[i]);
			if(count == null) {
				histogram.put((Integer)clusterSizes[i],1);
			}else {
				histogram.put((Integer)clusterSizes[i],count+1);
			}
			if(clusterSizes[i]>biggest) {
				biggest = clusterSizes[i];
				biggest_pos = i;
			}
		}
		System.out.println("" + empty.size() + " empty clusters");
		System.out.println("" + big.size() + " clusters with >= 5");
		System.out.println("Histogram");
		for(int i=0;i<=biggest;i++) {
			Integer count = histogram.get((Integer) i);
			if(count == null) {
				count = 0;
			}
			System.out.println("" + i + ": " + count);
		}

		
		for(int j=0;j<big.size();j++)
		{
			Double mh = 0.0;
			Double mh13 = 0.0;
			Double mk = 0.0;
			Double mk13 = 0.0;
			Double mn = 0.0;
			Double mn13 = 0.0;
			ArrayList<Integer> peaksInBiggest = new ArrayList<Integer>();
			System.out.println();
			System.out.println();
			System.out.println("Cluster " + big.get(j));
			for(int i=0;i<data.numPeaksets;i++){
				if(z[i].equals(big.get(j))) {
					peaksInBiggest.add(i);
				}
			}
			for(int i=0;i<peaksInBiggest.size();i++) {
				int peak = peaksInBiggest.get(i);
				System.out.println("Mass: " + data.masses[0][peak] + " RT: " + data.retentionTimes[0][peak] + " Transformation: " + transformations.get(tr[peak]).getName() + " transformed mass: " + transformed_mass[peak]);
				if(transformations.get(tr[peak]).getName().equals("M+H")) {
					mh = (Double)data.intensities[0][peak];
				}
				if(transformations.get(tr[peak]).getName().equals("M+HC13")) {
					mh13 = (Double)data.intensities[0][peak];
				}
				if(transformations.get(tr[peak]).getName().equals("M+K")) {
					mk = (Double)data.intensities[0][peak];
				}
				if(transformations.get(tr[peak]).getName().equals("M+KC13")) {
					mk13 = (Double)data.intensities[0][peak];
				}
				if(transformations.get(tr[peak]).getName().equals("M+Na")) {
					mn = (Double)data.intensities[0][peak];
				}
				if(transformations.get(tr[peak]).getName().equals("M+NaC13")) {
					mn13 = (Double)data.intensities[0][peak];
				}

			}
			Double hrat = mh/mh13;
			Double krat = mk/mk13;
			Double nrat = mn/mn13;
			System.out.println("M+H isotope ratio: " + hrat + "(" + mh + "/" + mh13 + ")");
			System.out.println("M+K isotope ratio: " + krat + "(" + mk + "/" + mk13 + ")");
			System.out.println("M+Na isotope ratio: " + nrat + "(" + mn + "/" + mn13 + ")");
		}

	}
	private void initialiseClustering() {
		Integer nClusters = data.numPeaksets;
		clusterSizes = new Integer[nClusters];
		clusterMassSums = new Double[nClusters];
		clusterRTSums = new Double[nClusters];
		clusterPriorMeanMass = new Double[nClusters];
		clusterPriorMeanRT = new Double[nClusters];
		z = new Integer[nClusters];
		tr = new Integer[nClusters];
		transformed_mass = new Double[nClusters];
		for(int i=0;i<nClusters;i++) {
			clusterSizes[i] = 1;
			clusterRTSums[i] = data.retentionTimes[0][i];
			clusterPriorMeanRT[i] = data.retentionTimes[0][i];
			ArrayList<Possible> a = possibles.get(i);
			z[i] = i;
			tr[i] = proton_pos;
			for(int j=0;j<a.size();j++) {
				if(a.get(j).cluster == i) {
					clusterMassSums[i] = a.get(j).mass;
					clusterPriorMeanMass[i] = a.get(j).mass;
					transformed_mass[i] = a.get(j).mass;
				}
			}
		}
	}
	private void findPossible()
	{
		possibles = new ArrayList<ArrayList<Possible>>();
		for(int i=0;i<data.numPeaksets;i++) {
			possibles.add(new ArrayList<Possible>());
		}
		for(int i=0;i<data.numPeaksets;i++) {
			// System.out.print("Using peak " + i + " as M+H");
			Double mh_mass = transformations.get(this.proton_pos).transformMass(data.masses[0][i]);
			// System.out.println(" Precursor mass = " + mh_mass);
			for(int j=0;j<data.numPeaksets;j++) {
				if(check(data.retentionTimes[0][i],data.retentionTimes[0][j])) {
					for(int k=0;k<transformations.size();k++) {
						Double t_mass = transformations.get(k).transformMass(data.masses[0][j]);
						if(check(mh_mass,t_mass,data.retentionTimes[0][i],data.retentionTimes[0][j])){
							possibles.get(j).add(new Possible(i,t_mass,k));
						}
					}
				}
			}
		}
	}
	private Boolean check(Double rt1,Double rt2) {
		if(Math.abs(rt1-rt2) > rt_tol) {
			return false;
		}else {
			return true;
		}
	}
	private Boolean check(Double mass1,Double mass2,Double rt1,Double rt2) {
		if(Math.abs(rt1-rt2) > rt_tol) {
			return false;
		}
		if(Math.abs(mass1-mass2)/mass1 > mass_tol/1e6) {
			return false;
		}
		return true;
	}
	private void loadTransformations() {
		String fileName = "mulsub2.txt";
		transformations = new Vector<Transformation>();
		BufferedReader fileReader;
		try {
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			while((line = fileReader.readLine())!=null) {
				String[] splitLine = line.split(",");
				String name = splitLine[0];
				Double sub = Double.parseDouble(splitLine[1]);
				Double mul = Double.parseDouble(splitLine[2]);
				Double iso = Double.parseDouble(splitLine[3]);
				transformations.add(new Transformation(name,sub,mul,iso));
				System.out.println(transformations.get(transformations.size()-1));
			}
			fileReader.close();
		}catch(IOException e) {
			System.out.println(e);
		}
	}
	private class Possible {
		public Integer cluster;
		public Double mass;
		public Integer transformation;
		public Possible(Integer cluster,Double mass,Integer transformation) {
			this.cluster = cluster;
			this.mass = mass;
			this.transformation = transformation;
		}
		public String toString() {
			return "" + cluster + ", " + mass + ", " + transformations.get(transformation).getName();
		}
 	}
	public static void main(String[] args) {
		new Group();
	}
}

// public class playing {
// 	public static void main(String[] args) throws Exception{
// 		String infile = "/Users/simon/Dropbox/BioResearch/Meta_clustering/JoeDataProcessing/Standards/std1 pos/std1-file1.peakml"; 
// 		// File input = new File(infile);
// 		ParseResult result = PeakMLParser.parse(new FileInputStream(infile), true);
// 		IPeakSet<IPeak> peaks = (IPeakSet<IPeak>) result.measurement;
// 		for(IPeak ps : peaks) {
// 			System.out.println("" + ps.getMass() + "," + ps.getRetentionTime());
// 		}
// 		// Data myData = new Data(result.header,peaks);
// 		// for(int i=0;i<myData.numReplicates;i++) {
// 		// 	System.out.println(myData.masses[i][0]);
// 		// }

// 	}
// }