package test;

import java.util.ArrayList;

public interface AccuracyFunction {
		//Compute Variance
		public float computeVar();
		
		//Compute Real Error 
		public float computeRealError();

		//Compute Error Classes (0-1)
		public int[] computeSimpleErrorClasses();
		
		
		//Print error Classes from 0.0 to 1.0
		public String printSimpleErrorClasses();
		
		
		public ArrayList<Float> getResult();
		public ArrayList<Float> getExpected();
		
		public void setResult(ArrayList<Float> result);
		public void setExpected(ArrayList<Float> expected);
		
}
