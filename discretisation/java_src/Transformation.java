public class Transformation {
	private String name;
	private Double mul,sub,iso;
	public Transformation(String name,Double sub,Double mul,Double iso) {
		this.name = name;
		this.mul = mul;
		this.sub = sub;
		this.iso = iso;
	}
	public Double transformMass(Double mass) {
		return (mass - this.sub)/this.mul + this.iso;
	}
	public String getName() {
		return this.name;
	}
	public String toString() {
		return this.name + "(" + this.sub + "," + this.mul + "," + this.iso + ")";
	}
}