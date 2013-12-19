
public class Peak {
	private Double mass;
	private Double intensity;
	private Double rt;
	private Double prob;
	private String databaseID;
	private String adduct;
	private String notation;
	public Peak(Double m,Double r,Double i,String d,String a,String n,Double p) {
		this.mass = m;
		this.intensity = i;
		this.rt = r;
		this.prob = p;
		this.databaseID = d;
		this.adduct = a;
		this.notation = n;
	}
	public boolean allS32() {
		int i = notation.indexOf("[34S]");
		if(i==-1) {
			return true;
		} else {
			return false;
		}

	}
	public boolean allN14() {
		int i = notation.indexOf("[15N]");
		if(i==-1) {
			return true;
		} else {
			return false;
		}
		
	}
	public boolean allC12() {
		int i = notation.indexOf("[13C]");
		if(i==-1) {
			return true;
		} else {
			return false;
		}
	}
	public boolean allO16() {
		int i = notation.indexOf("[18O]");
		if(i==-1) {
			return true;
		} else {
			return false;
		}
	}
	public boolean peakEquals(Peak p) {
		if(this.databaseID.equals(p.getDatabaseID()) && this.adduct.equals(p.getAdduct()) && this.notation.equals(p.getNotation())) {
			return true;
		} else {
			return false;
		}
	}
	public Double getIntensity() {
		return intensity;
	}
	public String getDatabaseID() {
		return databaseID;
	}

	public String getAdduct() {
		return adduct;
	}

	public void setAdduct(String adduct) {
		this.adduct = adduct;
	}

	public String getNotation() {
		return notation;
	}

	public void setNotation(String notation) {
		this.notation = notation;
	}

	public void setDatabaseID(String databaseID) {
		this.databaseID = databaseID;
	}
	
	
}
