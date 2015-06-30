
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
	private Double clusterPriorPrecisionMass = 1.0;
	private Double precisionRT = 1.0;
	private Double precisionMass = 1.0;
	private Integer[] z;
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
	}

	private void initialiseClustering() {
		Integer nClusters = data.numPeaksets;
		clusterSizes = new Integer[nClusters];
		clusterMassSums = new Double[nClusters];
		clusterRTSums = new Double[nClusters];
		clusterPriorMeanMass = new Double[nClusters];
		clusterPriorMeanRT = new Double[nClusters];
		z = new Integer[nClusters];
		for(int i=0;i<nClusters;i++) {
			clusterSizes[i] = 1;
			clusterRTSums[i] = data.retentionTimes[0][i];
			clusterPriorMeanRT[i] = data.retentionTimes[0][i];
			ArrayList<Possible> a = possibles.get(i);
			z[i] = i;
			for(int j=0;j<a.size();j++) {
				if(a.get(j).cluster == i) {
					clusterMassSums[i] = a.get(j).mass;
					clusterPriorMeanMass[i] = a.get(j).mass;
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
				for(int k=0;k<transformations.size();k++) {
					Double t_mass = transformations.get(k).transformMass(data.masses[0][j]);
					if(check(mh_mass,t_mass,data.retentionTimes[0][i],data.retentionTimes[0][j])){
						possibles.get(j).add(new Possible(i,t_mass,k));
					}
				}
			}
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