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

public class GetSignals {
	private String inFile;
	private String outFile;
	// private String inFile = "test.peakml";
	// private String inFile = "test2.peakml";
	private ParseResult result;
	private Data data;
	private Vector<Transformation> transformations;
	private static final Double mass_tol = 5.0;
	private static final Double rt_tol = 10.0;
	private int proton_pos;
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
	public GetSignals(String infile,String outfile) {
		this.inFile = infile;
		this.outFile = outfile;
		try {
			result = PeakMLParser.parse(new FileInputStream(inFile),true);
			data = new Data(result.header,(IPeakSet<IPeak>) result.measurement);
			// Unlog masses 
			for(int i=0;i<data.numPeaksets;i++) {
				for(int j=0;j<data.numReplicates;j++) {
					data.masses[j][i] = Math.exp(data.masses[j][i]);
				}
			}
			// System.out.println("Loaded " + data.numPeaksets + " peaksets");
		}catch(Exception e) {
			System.out.println(e);
		}
		// System.out.println(data.signals[0][546]);
		// System.out.println(data.retentionTimes[0][546]);
		// System.out.println(data.intensities[0][546]);

		try {
			PrintWriter writer = new PrintWriter(outfile, "UTF-8");
			for(int i=0;i<data.numPeaksets;i++) {
				String shape = "";
				int sl = data.signals[0][i].getSize();
				double[] x = data.signals[0][i].getX();
				double[] y = data.signals[0][i].getY();
				for(int j=0;j<sl;j++){
					shape += "" + x[j] + ":" + y[j] + " ";
				}
				System.out.println("" + data.masses[0][i] + "," + data.retentionTimes[0][i] + "," + data.intensities[0][i] + "," + shape); 
				writer.println("" + data.masses[0][i] + "," + data.retentionTimes[0][i] + "," + data.intensities[0][i] + "," + shape);
			}
			}catch(Exception e){
				System.out.println(e);
			}

	}

	public static void main(String[] args) {
		new GetSignals(args[0],args[1]);
	}
}