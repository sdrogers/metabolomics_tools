import java.io.*;
import java.util.*;

public class MolDB {
	private final String fileName = "std1_mols.csv";
	public ArrayList<Molecule> mols = new ArrayList<Molecule>();
	public MolDB() {
		loadMols();
	}
	private void loadMols() {
		BufferedReader inFile;
		try {
			inFile = new BufferedReader(new FileReader(fileName));
			String line;
			while((line = inFile.readLine())!=null){
				String[] splitLine = line.split(",");
				mols.add(new Molecule(splitLine[1],Double.parseDouble(splitLine[3]),splitLine[2]));
			}
			inFile.close();
		}catch(IOException e){
			System.out.println(e);
		}		
	}

	public ArrayList<Molecule> getHits(Double mass, Double tol) {
		ArrayList<Molecule> hits = new ArrayList<Molecule>();
		for(Molecule m: mols) {
			if(checkMass(mass,m.mass,tol)) {
				hits.add(m);
			}
		}
		return hits;
	} 
	public static boolean checkMass(Double mass1, Double mass2, Double tol) {
		if(Math.abs(mass1-mass2)/mass1 > tol/1e6) {
			return false;
		}
		else {
			return true;
		}
	}
	public static void main(String[] args) {
		MolDB m = new MolDB();
		// Just testing getCarbon
		System.out.println(m.mols.get(21).formula + " " + m.mols.get(21).getCarbon());
	}
}