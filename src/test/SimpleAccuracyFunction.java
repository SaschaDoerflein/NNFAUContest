package test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

public class SimpleAccuracyFunction implements AccuracyFunction{
	//First expected, second neural network result
	private ArrayList<Float> result;
	private ArrayList<Float> expected;
	
	public SimpleAccuracyFunction() {
		result = new ArrayList<Float>();
		expected = new ArrayList<Float>();
	}
	
	//Compute Variance
	public float computeVar() {
		float var = 0;

		 for(int i = 0; i<result.size();i++) {

			 float diff = result.get(i)-expected.get(i);
			 diff = Math.abs(diff);
			 diff = diff *10;
			 diff = diff * diff;
		     var += diff;
		    }
	
		
		return var/(float)result.size();
	}
	
	//Compute Real Error 
	public float computeRealError() {
		float rError=0;
		
		for(int i=0; i<result.size();i++) {
			float diff = result.get(i)-expected.get(i);
			 diff = Math.abs(diff);
			 rError += diff;
		}
		
		return rError;
	}
	
	//Compute Error Classes (0-1)
	public int[] computeSimpleErrorClasses() {
		float[] resultArray = new float[result.size()];
		int[] errorClasses = new int[10];
		
		for(int i=0; i<result.size();i++) {
			float diff = result.get(i)-expected.get(i);
			 diff = Math.abs(diff);
			 resultArray[i] = diff;
		}
		Arrays.sort(resultArray);
	
		//Normalisation
		float max = resultArray[resultArray.length-1];
		
		for(int i = 0; i<result.size(); i++) {
			float normalValue = resultArray[i]/max;
			normalValue = normalValue*10;
			int index = Math.round(normalValue);
			if (index > 9) {
				index = 9;
			}
			errorClasses[index] = errorClasses[index] + 1;
		}
		
		return errorClasses;
	}
	
	public String printSimpleErrorClasses() {
		int[] errorClasses = computeSimpleErrorClasses();
		String output = "";
		int overallSum = 0;
		int halfSum = 0;
	
		boolean foundMedian = false;
		System.out.println("Differenz classes");
		float index=0;
		for(float entry : errorClasses) {
			int realEntry = (int)(entry);
			float prozentEntry = entry/(float)result.size();

			prozentEntry *= 1000;
			prozentEntry = Math.round(prozentEntry);
			prozentEntry /= 10;
			
			overallSum += realEntry;
			
			System.out.print(index/10.0+": "+realEntry+" "+prozentEntry+"%");
			if(index>=5) {
				halfSum += realEntry;
			}
			
			if(foundMedian == false && overallSum >= (float)result.size()/(float)2 ) {
				System.out.println("<--");
				foundMedian = true;
			}else {
				System.out.println("");
			}
			index += 1;
					}
		
		System.out.println("Error till 0.5: "+halfSum+" "+halfSum/(float)result.size()+"%");
		
		return output;
	}
	
	

	@Override
	public ArrayList<Float> getResult() {
		
		return result;
	}

	@Override
	public ArrayList<Float> getExpected() {
		
		return expected;
	}

	@Override
	public void setResult(ArrayList<Float> result) {
		this.result= result;
		
	}

	@Override
	public void setExpected(ArrayList<Float> expected) {
		this.expected = expected;
		
	}
	
}
