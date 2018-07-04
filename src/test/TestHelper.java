package test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import neuralNetwork.Blob;
import neuralNetwork.DataReader;
import neuralNetwork.Datum;
import neuralNetwork.Network;

public class TestHelper {
	
	AccuracyFunction af;
	
	public TestHelper(AccuracyFunction af) {
		this.af = af;
	}
	
	//Write every result from every iterations separated in epochs into a output file
	public void performTest(Network[] networks, int iterations,String fileName, int option) {
		System.out.println("------------"+fileName+"------------");
		float[] realErrors = new float[networks.length];
		float[] variances = new float[networks.length];
		int[][] errorClassesArray = new int[networks.length][];
		
		
			//Data 	
			ArrayList<Datum> dataAndLabel=null;
			try {
				if (option == 1) {
					dataAndLabel = DataReader.readTitanicDataset(fileName,true);
				}else if(option == 2) {
					dataAndLabel = DataReader.readTraveltimeDataset(fileName,true);
				}else if (option == 3) {
					dataAndLabel = DataReader.getImageDataset(fileName,true);
				}
				
			
			int length = dataAndLabel.size();
			
			
			
			//Normalisation
			//TODO: Sascha 4.7.  Should be good but makes thinks worse :(
			
			int lenOfBlob = dataAndLabel.get(0).data.values.length;	
			float[] maxes = new float[lenOfBlob];
			
			for(int i = 0; i < length; i++) {
				Blob b = dataAndLabel.get(i).data;
				
				for(int n = 0; n < b.values.length; n++) {
					if(maxes[n] < b.values[n]) {
						maxes[n] = b.values[n];
					}
				}
			}
			for(int i = 0; i < length; i++) {
				Blob b = dataAndLabel.get(i).data;
				
				for(int n = 0; n < b.values.length; n++) {
					b.values[n] = b.values[n]/maxes[n];
				}
			}
			
			
			//Iteration
			for(int e=0; e<networks.length; e++) {
			
			
			ArrayList<Float> result = af.getResult();
			ArrayList<Float> expected = af.getExpected();
			
			for(int j=0;j<iterations;j++)
			{
			if (j%100==0) System.out.println("Iteration: "+j);

				for(int i=0;i<length;i++)
				{
					int idx=i;
					Blob out=networks[e].trainSimpleSGD(dataAndLabel.get(idx).data, dataAndLabel.get(idx).label);

					if((j==iterations-1 || j<2) && i<10)
					{

						for(int h=0;h<out.getLength();h++)
						{
						 	//System.out.print(out.getValue(h)+" ");
						 	result.add(out.getValue(h)*maxes[h]);
						}

						//System.out.print("vs. ");

						for(int h=0;h<dataAndLabel.get(idx).label.getLength();h++)
						{
						 	//System.out.print(dataAndLabel.get(idx).label.getValue(h)+" ");
						 	expected.add(dataAndLabel.get(idx).label.getValue(h));
						}

						
						//System.out.println();
					}
				}
			}
			float realError = af.computeRealError();
			float variance = af.computeVar();
			int[] errorClasses = af.computeSimpleErrorClasses();
			
			realErrors[e] = realError;
			variances[e] = variance;
			errorClassesArray[e] = errorClasses;
			
			System.out.println(System.lineSeparator());
			System.out.println("Epoch: "+e);
			System.out.println("Total diff:"+realError);
			
			System.out.println("Varianz of all results and expected Values: "+variance);
			
			af.printSimpleErrorClasses();
			af.setExpected(new ArrayList<Float>());
			af.setResult(new ArrayList<Float>());
			System.out.println(System.lineSeparator());
			
		
		
		}
			} catch (IOException ee) {
				// TODO Auto-generated catch block
				ee.printStackTrace();
			}
		
		/*
		 * TODO: For test file? We can't compute any variance because we have no expected values!
		 * 
		 */
	}

	
	

}
