
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

import java.util.zip.GZIPOutputStream;


public class Group {
	private String inFile = "std1-file1.peakml";
	// private String inFile = "test.peakml";
	// private String inFile = "test2.peakml";
	private ParseResult result;
	private Data data;
	private Vector<Transformation> transformations;
	private static final Double mass_tol = 5.0;
	private static final Double rt_tol = 10.0;
	private int proton_pos;
	private ArrayList<ArrayList<Possible>> possibles;
	private ArrayList<Integer> needUpdates;
	private Integer[] clusterSizes;
	private Double[] clusterMassSums; 
	private Double[] clusterRTSums;
	private Double[] clusterPriorMeanRT;
	private Double[] clusterPriorMeanMass;
	private Double clusterPriorPrecisionRT = 1.0;
	private Double[] clusterPriorPrecisionMass;
	private Double precisionRT = 1.0;
	private Double[] precisionMass;
	private Double ppm = 5.0;
	private Double alpha = 1.0;
	private Integer[] z;
	private Integer[] tr;
	private Double[] transformed_mass;
	private Random r;
	private int nSamples = 0;
	private int nBurn = 50;
	private int[] mapCluster;
	private int[] mapClusterSizes;
	private int[] mapTr;
	private Double[] mapProb;
	private MolDB molDB = new MolDB();
	private ArrayList<ArrayList<Molecule>> clusterHits;
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
		int i=0;
		Vector<IPeakSet<IPeak>> l = new Vector<IPeakSet<IPeak>>();
		for(IPeak p: (IPeakSet<IPeak>)result.measurement) {
			System.out.println("" + p.getMass() + ": " + data.masses[0][i] + " " + mapCluster[i] + " " + transformations.get(mapTr[i]).getName());
			IPeakSet<IPeak> q = new IPeakSet<IPeak>(p);
			q.addAnnotation(Annotation.relationship,transformations.get(mapTr[i]).getName());
			q.addAnnotation(Annotation.relationid,mapCluster[i]);

			i++;
			l.add(q);
		}
		IPeakSet<IPeakSet<IPeak>> peaks = new IPeakSet<IPeakSet<IPeak>>(l);

		try {
		PeakMLWriter.write(
					result.header, peaks.getPeaks(), null,
					new GZIPOutputStream(new FileOutputStream("test.peakml")), null
				);
		}catch(IOException e) {
			System.out.println(e);
		}
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
				precision = clusterPriorPrecisionMass[this_cluster] + precisionMass[this_cluster] * clusterSizes[this_cluster];
				mean = (1.0/precision)*(clusterPriorMeanMass[this_cluster]*clusterPriorPrecisionMass[this_cluster] + precisionMass[this_cluster]*clusterMassSums[this_cluster]);
				pred_precision = 1.0/(1.0/precision + 1.0/precisionMass[this_cluster]);
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
			if(this.nSamples > this.nBurn)
			{
				p.get(new_cluster).count ++;
			}
			clusterSizes[z[peak]] ++;
			clusterRTSums[z[peak]] += data.retentionTimes[0][peak];
			clusterMassSums[z[peak]] += transformed_mass[peak];

		}
		this.nSamples++;
	}
	private void summarise() {
		// Displaying clusters with >5 members
		ArrayList<Integer> big = new ArrayList<Integer>();
		ArrayList<Integer> empty = new ArrayList<Integer>();
		HashMap<Integer,Integer> histogram = new HashMap<Integer,Integer>();
		mapCluster = new int[data.numPeaksets];
		mapClusterSizes = new int[data.numPeaksets];
		mapTr = new int[data.numPeaksets];
		mapProb = new Double[data.numPeaksets];
		for(int i=0;i<data.numPeaksets;i++) {
			int nPossible = possibles.get(i).size();
			if(nPossible == 1) {
				possibles.get(i).get(0).count = this.nSamples;
				possibles.get(i).get(0).posteriorProb = 1.0;
				mapCluster[i] = possibles.get(i).get(0).cluster;
				mapTr[i] = possibles.get(i).get(0).transformation;
				mapProb[i] = 1.0;
			}else {
				Double maxPost = 0.0;
				mapCluster[i] = -1;
				for(Possible p: possibles.get(i)) {
					p.posteriorProb = (1.0*p.count)/(1.0*(this.nSamples-this.nBurn));
					if(p.posteriorProb >= maxPost) {
						maxPost = p.posteriorProb;
						mapCluster[i] = p.cluster;
						mapTr[i] = p.transformation;
						mapProb[i] = p.posteriorProb;
					}
				}
			}
			System.out.println(mapCluster[i]);
			mapClusterSizes[mapCluster[i]] += 1;
		}
		Integer nClusters = data.numPeaksets;
		Integer biggest = 0;
		int biggest_pos = -1;
		for(int i=0;i<nClusters;i++) {
			if(mapClusterSizes[i]>=4) {
				big.add(i);
			}
			if(mapClusterSizes[i]==0) {
				empty.add(i);
			}
			Integer count = histogram.get((Integer)mapClusterSizes[i]);
			if(count == null) {
				histogram.put((Integer)mapClusterSizes[i],1);
			}else {
				histogram.put((Integer)mapClusterSizes[i],count+1);
			}
			if(mapClusterSizes[i]>biggest) {
				biggest = mapClusterSizes[i];
				biggest_pos = i;
			}
		}
		System.out.println("" + empty.size() + " empty clusters");
		System.out.println("" + big.size() + " clusters with >= 4");
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
			summariseCluster(big.get(j));
		}

	}
	private void summariseCluster(int clustNo) {
			Double mh = 0.0;
			Double mh13 = 0.0;
			Double mk = 0.0;
			Double mk13 = 0.0;
			Double mn = 0.0;
			Double mn13 = 0.0;
			ArrayList<Integer> peaksInBiggest = new ArrayList<Integer>();
			System.out.println();
			System.out.println();
			System.out.println("Cluster " + clustNo);
			System.out.println("Hits: ");
			for(Molecule m: clusterHits.get(clustNo)) {
				System.out.print(m);
				System.out.println(" Cratio: " + m.cRatio());
			}
			for(int i=0;i<data.numPeaksets;i++){
				if(mapCluster[i] == clustNo) {
					peaksInBiggest.add(i);
				}
			}
			for(int i=0;i<peaksInBiggest.size();i++) {
				int peak = peaksInBiggest.get(i);
				System.out.println("Peak: " + peak + " Mass: " + data.masses[0][peak] + " RT: " + data.retentionTimes[0][peak] + " Intensity: " + data.intensities[0][peak] + " Transformation: " + transformations.get(mapTr[peak]).getName() + " transformed mass: " + transformations.get(mapTr[peak]).transformMass(data.masses[0][peak]) + " Probability: " + mapProb[peak]);
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
	private void initialiseClustering() {
		this.nSamples = 0;
		Integer nClusters = data.numPeaksets;
		clusterSizes = new Integer[nClusters];
		clusterMassSums = new Double[nClusters];
		clusterRTSums = new Double[nClusters];
		clusterPriorMeanMass = new Double[nClusters];
		clusterPriorMeanRT = new Double[nClusters];
		clusterPriorPrecisionMass = new Double[nClusters];
		precisionMass = new Double[nClusters];
		z = new Integer[nClusters];
		tr = new Integer[nClusters];
		transformed_mass = new Double[nClusters];
		clusterHits = new ArrayList<ArrayList<Molecule>>();
		for(int i=0;i<nClusters;i++) {
			clusterSizes[i] = 1;
			clusterRTSums[i] = data.retentionTimes[0][i];
			clusterPriorMeanRT[i] = data.retentionTimes[0][i];
			ArrayList<Possible> a = possibles.get(i);
			z[i] = i;
			tr[i] = proton_pos;
			for(int j=0;j<a.size();j++) {
				a.get(j).count = 0;
				if(a.get(j).cluster == i) {
					clusterMassSums[i] = a.get(j).mass;
					clusterPriorMeanMass[i] = a.get(j).mass;
					transformed_mass[i] = a.get(j).mass;
					clusterPriorPrecisionMass[i] = compPrecision(ppm,transformed_mass[i]);
					precisionMass[i] = compPrecision(ppm,transformed_mass[i]);
					if(molDB != null) {
						// Could do this afterwards with the posterior mass value?
						clusterHits.add(molDB.getHits(transformed_mass[i],mass_tol));
					}
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
					if(data.intensities[0][j]<=data.intensities[0][i]) {
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
		public Integer count;
		public Double posteriorProb;
		public Possible(Integer cluster,Double mass,Integer transformation) {
			this.cluster = cluster;
			this.mass = mass;
			this.transformation = transformation;
			this.count = 0;
			this.posteriorProb = 0.0;
		}
		public String toString() {
			return "" + cluster + ", " + mass + ", " + transformations.get(transformation).getName() + ", " + this.count + ", " + this.posteriorProb;
		}
 	}

 	public static Double compPrecision(double ppm,double mass) {
 		// Set 3 standard dev = ppm value
 		Double di = ppm*mass/1e6;
 		Double sd = di/3.0;
 		return 1.0/(sd*sd);
 	}

 	public static Double cRatios() {
 		Double p12 = 0.9893;
 		Double p13 = 0.0107;
 		Double pr = 1.0;
 		for(int i=1;i<20;i++) {
 			// n choose 1 is n
 			Double prob12 = pr*p12;
 			Double prob13 = pr*p13*i;
 			pr*=p12;
 			System.out.println("C = " + i + " rat = " + prob12/prob13);
 		}
 		return 1.0;
 	}
 	public static int factorial(int x) {
 		if(x==0) {
 			return 1;
 		}else {
 			return x*factorial(x-1);
 		}
 	}

 	public static int choose(int n,int k) {
 		return factorial(n)/(factorial(k)*factorial(n-k));
 	}

	public static void main(String[] args) {
		new Group();
	}
}

