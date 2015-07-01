/* Copyright (C) 2008, Groningen Bioinformatics Centre (http://gbic.biol.rug.nl/)
 * This file is part of mzmatch.
 * 
 * PeakML is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 * 
 * mzmatch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with PeakML; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */



// package mzmatch.ipeak.convert;


// java
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





/* HISTORY:
 * 1.0.1	- fixed a bug in the conversion of an id to a real name.
 */
public class ConvertToText
{
	// implementation
	public static void process(Header header, IPeakSet<IPeakSet<IPeak>> peaksets, Options options, HashMap<String,Molecule> molecules, OutputStream output) throws XmlParserException,IOException
	{
		final CsvWriter writer = new CsvWriter(output, '\t', Charset.forName("UTF-8"));
		writer.write("id");
		writer.write("mass");
		writer.write("RT");
		final List<Integer> measurements = new ArrayList<Integer>();
		for (MeasurementInfo measurement : header.getMeasurementInfos()) {
			writer.write(measurement.getLabel());
			measurements.add(measurement.getID());
		}
		for ( int i = 0; i < options.annotations.size(); ++i ) {
			writer.write(options.annotations.get(i));
		}
		writer.endRecord();
		
		for (int i=0; i<peaksets.size(); ++i)
		{
			IPeakSet<IPeak> peakset = peaksets.get(i);
			writer.write(peakset.getAnnotation("id").getValue());
			writer.write(Double.toString(peakset.getMass()));
			writer.write(Double.toString(peakset.getRetentionTime()));
			for ( int measurement : measurements ) {
				boolean written = false;
				for ( IPeak peak : peakset) {
					if ( peak.getMeasurementID() == measurement ) {
						written = true;
						writer.write(Double.toString(peak.getIntensity()), true);
					}
				}
				if ( ! written ) {
					writer.write(" ", false);
				}
			}
			for ( int j = 0; j < options.annotations.size(); ++j ) {
				final Annotation a = peakset.getAnnotation(options.annotations.get(j));
				if ( a != null ) {
					writer.write(a.getValue());
				} else {
					writer.write(" ", false);
				}
			}
			writer.endRecord();
		}
		writer.flush();
		writer.close();
		/*
		
		
		
		
		
		
		// create a vector out of all the measurement-ids
		Vector<String> measurements = new Vector<String>();
		measurements.add("mass");
		measurements.add("RT");	// room for the retention time
		for (MeasurementInfo measurement : header.getMeasurementInfos())
			measurements.add(measurement.getLabel());
		
		// create the matrix with the values
		DataFrame.Double df = new DataFrame.Double(peaksets.size(), measurements.size());
		df.setColNames(measurements);
		
		// retrieve the data
		Vector<HashMap<String,String>> annotations = new Vector<HashMap<String,String>>();
		for (int i=0; i<peaksets.size(); ++i)
		{
			IPeakSet<IPeak> peakset = peaksets.get(i);
			
			// retrieve the annotations
			HashMap<String,String> ann = new HashMap<String,String>();
			for (String annot : options.annotations)
				if (peakset.getAnnotation(annot) != null) {
					ann.put(annot, peakset.getAnnotation(annot).getValue());
				}
			annotations.add(ann);
			
			// fill the matrix
			for (IPeak peak : peakset)
			{
				int index = header.indexOfMeasurementInfo(peak.getMeasurementID());
				df.set(i, index+1, peak.getIntensity());
			}
			df.setRowName(i, peakset.getAnnotation("id").getValue());
			df.set(i, "mass", peakset.getMass());
			df.set(i, "RT", peakset.getRetentionTime());
		}

		
		// dump
		PrintStream out = new PrintStream(output);
		out.print("id\t");
		for ( int i = 0; i < df.getNrColumns(); ++i ) {
			out.print(df.getColName(i));
			if ( i < df.getNrColumns() - 1 || options.annotations.size() > 0 ) {
				out.print("\t");
			}
		}
		for ( int i = 0; i < options.annotations.size(); ++i ) {
			out.print(options.annotations.get(i));
			if ( i < options.annotations.size() - 1 ) {
				out.print("\t");
			}
		}
		
		out.println();
		
		Vector<String> rownames = new Vector<String>();
		for (int row=0; row<df.getNrRows(); ++row)
		{
			String name = df.getRowName(row);
			while (rownames.contains(name))
				name += '0';
			rownames.add(name);
			
			out.print(name);

			for (int col=0; col<df.getNrColumns(); ++col)
				out.print("\t" + df.get(row, col));
			HashMap<String,String> annot = annotations.elementAt(row);
			for (String n : options.annotations)
			{
				if (n.equals("identification") && annot.get(n)!=null)
				{
					String ids = "";
					for (String id : annot.get(n).split(","))
					{
						Molecule m = molecules.get(id.trim());
						if (ids.length() != 0)
							ids = ids + ", ";
						if (m == null)
							ids = ids + id;
						else
							ids = ids + m.getName().trim();
					}
					out.print("\t" + ids);
				}
				else
					out.print("\t" + (annot.get(n)==null ? "" : annot.get(n)));
			}
			out.println();
		}
		*/
	}
	
	
	// main entrance
	final static String version = "1.0.1";
	final static String application = "ConvertToText";
	@OptionsClass(name=application, version=version, author="RA Scheltema (r.a.scheltema@rug.nl)",
		description=
		"Converts the contents of a PeakML file to a tab-separated text file. In order to make " +
		"the file useable in a spreadsheet environment, only the top-level data for each measurement " +
		"is given in the file. This application has been extended with support for outputting " +
		"annotations.",
		example=
		"Windows batch-file:\n" +
		"SET JAVA=java -cp mzmatch.jar -da -dsa -Xmn1g -Xms1425m -Xmx1425m -Xss128k -XX:+UseParallelGC -XX:ParallelGCThreads=10\n" +
		"\n" +
		"REM extract all the mass chromatograms\n" +
		"%JAVA% mzmatch.ipeak.convert.ConvertToText -v -i file.peakml -o file.txt -annotations \"relation.id,charge\"",
		references=
		""
	)
	public static class Options
	{
		@Option(name="i", param="filename", type=Option.Type.REQUIRED_ARGUMENT, level=Option.Level.USER, usage=
			"The input file in the PeakML file format. When this option is not set, the input is " +
			"read from the standard input.")
		public Vector<String> input = new Vector<String>();
		@Option(name="o", param="filename", type=Option.Type.REQUIRED_ARGUMENT, level=Option.Level.USER, usage=
			"The output file in text format. When this option is not set, the ouput is written to the " +
			"standard out. Be sure to unset the verbose option when setting up a pipeline reading and " +
			"writing from the standard in- and outputs.")
		public String output = null;
		
		@Option(name="databases", param="filename", type=Option.Type.REQUIRED_ARGUMENT, level=Option.Level.USER, usage=
			"Optional list of database-files in the moleculedb format. These databases are used " +
			"to translate the id's in the annotation 'identifications' to full molecule names.")
		public Vector<String> databases = new Vector<String>();
		@Option(name="annotations", param="label", type=Option.Type.REQUIRED_ARGUMENT, level=Option.Level.USER, usage=
			"Optional list of annotations to be exported.")
		public Vector<String> annotations = new Vector<String>();

		@Option(name="h", param="", type=Option.Type.NO_ARGUMENT, level=Option.Level.SYSTEM, usage=
			"When this is set, the help is shown.")
		public boolean help = false;
		@Option(name="v", param="", type=Option.Type.NO_ARGUMENT, level=Option.Level.SYSTEM, usage=
			"When this is set, the progress is shown on the standard output.")
		public boolean verbose = false;
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String args[])
	{
		try
		{
			Tool.init();
			
			// parse the commandline options
			Options options = new Options();
			CmdLineParser cmdline = new CmdLineParser(options);
			
			// check whether we need to show the help
			cmdline.parse(args);
			if (options.help)
			{
				Tool.printHeader(System.out, application, version);
				
				cmdline.printUsage(System.out, "");
				return;
			}
			
			if (options.verbose)
			{
				Tool.printHeader(System.out, application, version);
				cmdline.printOptions();
			}
			
			// check the command-line parameters
			{
				// check whether the user gave a list of files or a single file
				if (options.input.size() == 0)
				{
					// we're inputting from stdin
					options.input.add(null);
				}
				else if (options.input.size() == 1)
				{
					// a single file
					File inputfile = new File(options.input.firstElement());
					if (!inputfile.exists())
					{
						System.err.println("[ERROR]: the input-file '" + options.input.firstElement() + "' does not exist.");
						System.exit(0);
					}
				}
				else
				{
					// multiple input files
					for (String filename : options.input)
					{
						File inputfile = new File(filename);
						if (!inputfile.exists())
						{
							System.err.println("[ERROR]: the input-file '" + filename + "' does not exist.");
							System.exit(0);
						}
					}
					
					// check whether we have an output destination
					if (options.output == null)
					{
						System.err.println("[ERROR]: multiple input files defined without an output destination.");
						System.exit(0);
					}
				}
				
				// if the output directories do not exist, create them
				if (options.output != null)
					Tool.createFilePath(options.output, options.input.size()==1);
			}
			
			
			// load the molecule data
			HashMap<String,Molecule> molecules = new HashMap<String,Molecule>();
			for (String filename : options.databases)
				molecules.putAll(MoleculeIO.parseXml(new FileInputStream(filename)));
			
			// start processing
			for (String filename : options.input)
			{
				// open the streams
				InputStream input = System.in;
				if (filename != null)
					input = new FileInputStream(filename);
				
				OutputStream out_output = System.out;
				if (options.input.size() == 1)
				{
					if (options.output != null)
						out_output = new FileOutputStream(options.output);
				}
				else
				{
					String name = new File(filename).getName();
					out_output = new FileOutputStream(options.output + "/" + name);
				}
				
				// load the data
				if (options.verbose)
					System.out.println("Loading data");
				ParseResult result = PeakMLParser.parse(input, true);
				IPeakSet<IPeak> peaks = (IPeakSet<IPeak>) result.measurement;
				IdentifyPeaksets.identify(peaks);
				// process and write
				if (options.verbose)
					System.out.println("Processing data");
				if (peaks.getContainerClass().equals(MassChromatogram.class))
				{
					;
				}
				else if (peaks.getContainerClass().equals(IPeakSet.class))
				{
					process(
							result.header,
							(IPeakSet<IPeakSet<IPeak>>) result.measurement,
							options,
							molecules,
							out_output
						);
				}
				else
					System.err.println("Unknown IPeak type: " + peaks.getContainerClass().getClass().getName());
			}
		}
		catch (Exception e)
		{
			Tool.unexpectedError(e, application);
		}
	}
}
