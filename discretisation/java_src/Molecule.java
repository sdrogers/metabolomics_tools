public class Molecule {
	public final String name;
	public final Double mass;
	public final String formula;
	public Molecule(String name, Double mass, String formula) {
		this.name = name;
		this.mass = mass;
		this.formula = formula;
	}
	public int getCarbon() {
		return getCarbon(this);
	}
	public String toString() {
		return name + "(" + formula + "): " + mass;
	}
	public Double cRatio() {
		int c = this.getCarbon();
		Double p12 = 0.9893;
 		Double p13 = 0.0107;
 		return Math.pow(p12,c)/(Math.pow(p12,c-1)*p13*c);
	}
	public static int getCarbon(Molecule m) {

		String formula = m.formula;
		return getCarbon(formula);
	}
	public static int getCarbon(String formula) {
		int startPos = 0;
		int endPos = 1;
		while(!formula.substring(startPos,startPos+1).equals("C")) {
			startPos ++;
		}
		startPos++;
		endPos = startPos;
		while(!formula.substring(endPos,endPos+1).equals("H")) {
			endPos++;
		}
		return Integer.parseInt(formula.substring(startPos,endPos));
	}
}