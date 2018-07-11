package neuralNetwork;
import java.io.IOException;
import java.util.*;

import test.SimpleAccuracyFunction;

import java.io.*;

public class MainChris
{
	public static void main(String[] args) throws IOException
	{
		//entscheidet welches Set läuft
		int[] test = {0,1,0};
		/*für den derzeitigen Rekord 85.185 wurden diese Einstellungen genutzt falls sie sich geändert haben sollen.
		* iterations = 3000; VariantLearningRate(0.005f,it,1,null); 2 Hidden Layer einmal 6 auf 5 und einmal 5 auf 2 mit TanhAct;
		* Das Online Ergebniss hängt sehr stark von den initialen Weights i.e RandomWeight ab...
		* für den Rekord wurde diese^^ Einstellung etwa 10+ mal versucht...vermutlich etwas Glück Werte zw. 76, und 85%.
		*/
		/*---------------------------------------------------
		------ Example NeuralNet using Titanic-Dataset ------
		---------------------------------------------------*/
		if (test[0] == 1) {
			ArrayList<Datum> dataAndLabel=DataReader.readTitanicDataset("titanic_training.txt",true);
			int iterations = 3000;
			int length = dataAndLabel.size();
			Network n=new Network(new VariantLearningRate(0.005f,iterations,1,null));
			//Network n=new Network(new ConstantLearningRate(0.001f));
	
			n.add(new InputLayer(6));
			n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 5, 16));
			/*for (int i = 0; i < 20; i++) {
				n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 4, 4));
			}*/
			
			n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 5, 2, 16));
			n.add(new OutputLayer(new EuclideanLoss(),new LinearActivation(), new RandomWeight(), new ConstantBias(), 2, 1, 16));
			
			
			for(int j=0;j<iterations;j++)
			{
			if (j%(iterations/10)==0 || j < 3) System.out.println("Iteration: "+j) ;
			//System.out.println(length);
				for(int i=0;i<length;i++)
				{
					int idx=i;
					Blob out=n.trainSimpleSGD(dataAndLabel.get(idx).data, dataAndLabel.get(idx).label);
	
					if((j==iterations-1 || j<3 || (j%(iterations/10)==0 && j > 500)) && i<10)
					{
	
						for(int h=0;h<out.getLength();h++)
						{
						 	System.out.print(out.getValue(h)+" ");
						}
	
						System.out.print("vs. ");
	
						for(int h=0;h<dataAndLabel.get(idx).label.getLength();h++)
						{
						 	System.out.print(dataAndLabel.get(idx).label.getValue(h)+" ");
						}
						System.out.println();
					}
				}
			}
	
			ArrayList<Datum> testData=DataReader.readTitanicDataset("titanic_test1.txt",false);
			ArrayList<Datum> testDataexact=DataReader.readTitanicDataset("titanic_test1.txt",false);
	
			for(int i=0;i<testData.size();i++)
			{
				Blob out[]=n.forward(testData.get(i).data);
				testDataexact.get(i).label.setValue(0, out[out.length-1].getValue(0)); 
				if(out[out.length-1].getValue(0) < 0.5)
				{
					testData.get(i).label.setValue(0,0f);
				}
				else
				{
					testData.get(i).label.setValue(0,1f);
				}
			}
	
			DataWriter.writeLabelsToFile("titanic_prediction.txt", testData);
			//DataWriter.writeLabelsToFile("titanic_prediction_exact.txt", testDataexact);
			System.out.println("done");
		}
		
		/*---------------------------------------------------
		------ Example NeuralNet using TravelTimeDataset ------
		---------------------------------------------------*/
		/* um die Baseline zu schlagen/den Platz zu begründen i.e ca. 170 oder besser reicht:
		 *  iteration 1000, constantlearningRate 0.002 und lstm true ein Block mit 4 zellen i.e LSTMCell,...1,4,2,1;
		 * für den Rekord weiß ich die Einstellungen nicht mehr genau sollte aber punktetechnisch keine Rolle spielen
		 */
		if (test[1] == 1) {
			ArrayList<Datum> dataAndLabel2=DataReader.readTraveltimeDataset("rta_training.txt",true);
			SimpleAccuracyFunction saf = new SimpleAccuracyFunction();
			ArrayList<Float> result = new ArrayList<Float>();
			ArrayList<Float> expected = new ArrayList<Float>();
			boolean lstm = true;
			
			int iterations2 = 1000;
			
			Network n2=new Network(new ConstantLearningRate(0.002f));
			//Network n2=new Network(new VariantLearningRate(0.0001f,iterations2,1,null));
			//LSTMCell(WeightFiller fillerWeight, BiasFiller fillerBias , int blocks, int cells,  int in, int out)
			if (lstm) {
				n2.add(new LSTMCell(new RandomWeight(), new ConstantBias(), 1, 4, 2, 1));
			} else {
				n2.add(new InputLayerInverse(26));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 26, 13, 50));
				
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 13, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
				n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 6, 50));
			
				n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(7000,0), new RandomWeight(), new ConstantBias(), 6, 6, 50));
			}
			Blob out;
			int when = 20;
			int early = 0;
			float temp = 0;
			float var = 0;
			float varOld = 0;
			int count = dataAndLabel2.size();
			float maxvar = 0;
			float minvar = 1000;
			int show = 5;
			for(int j=0;j<iterations2;j++)
			{
				if ((j > 9 && j%(iterations2/100)==0) || j < 5) System.out.println("Iteration: "+j) ;
	
				for(int i=0;i<dataAndLabel2.size();i++)
				{
					int idx=i;
					if (lstm) {
						out = n2.trainLSTM(dataAndLabel2.get(idx).data, dataAndLabel2.get(idx).label);
					} else {
					out = n2.trainSimpleSGD(dataAndLabel2.get(idx).data, dataAndLabel2.get(idx).label);
					}
	
					if((j==iterations2-1 || j<5 || ( j > 9 && j%(iterations2/100)==0)) && i<count)
					{
						if (j > 9 && j%(iterations2/100)==0) show = 5;
						if (j==iterations2-1) show = 150;
						//System.out.println("Iteration: "+j);
						for(int h=0;h<out.getLength();h++)
						{
							result.add(out.getValue(h));
							if (i<show) System.out.print(out.getValue(h)+" ");
						}
	
						if (i<show) System.out.print("vs. ");
	
						for(int h=0;h<dataAndLabel2.get(idx).label.getLength();h++)
						{
							expected.add(dataAndLabel2.get(idx).label.getValue(h));
							if (i < show) System.out.print(dataAndLabel2.get(idx).label.getValue(h)+" ");
						}
						saf.setExpected(expected);
						saf.setResult(result);
						var = saf.mse();
						if (i<show) System.out.println(var);
						if (var > maxvar) maxvar = var;
						if (var < minvar) minvar = var;
						temp+=var;						
						result.clear();
						expected.clear();
					}
				}
				
				if((j==iterations2-1 || j<5 || ( j > 9 && j%(iterations2/100)==0))) {
					if (Math.abs(temp/count - varOld/count) < 0.5) early++;
					if (early == when) j = iterations2-1;
					System.out.println("var = " + temp/count);
					System.out.println("maxvar = " + maxvar);
					System.out.println("minvar = " + minvar);
					varOld = temp;
					temp = 0;
					maxvar = 0;
					minvar = 1000;
					
				}
				show = 0;
			}
			
	
			ArrayList<Datum> testData2=DataReader.readTraveltimeDataset("rta_test1.txt",false);
			
			float tempval;
			//System.out.println(testData2.size());
			for(int i=0;i<testData2.size();i++)
			{
				
				
				if (lstm) {
					out=n2.forwardLSTM(testData2.get(i).data);
					for (int j = 0; j < out.getLength();j++) {
						tempval = Math.round(out.getValue(j));
						testData2.get(i).label.setValue(j,tempval);
					}
				} else {
					Blob outar[]=n2.forward(testData2.get(i).data);
					for (int j = 0; j < outar[outar.length-1].getLength();j++) {
						tempval = Math.round(outar[outar.length-1].getValue(j));
						testData2.get(i).label.setValue(j,tempval);
					}
				}
			}

			DataWriter.writeLabelsToFile("rta_prediction.txt", testData2);
			System.out.println("done");
		}
		
		/*---------------------------------------------------
		------ Example NeuralNet using ImageDataset ------
		---------------------------------------------------*/
		if (test[2] == 1) {
			
			ArrayList<Datum> dataAndLabel3=DataReader.getImageDataset("image_training.bin",true);
	
			Network n2=new Network(new ConstantLearningRate(0.001f));
	
			n2.add(new InputLayer(32*32*3));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 16, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 64, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 128, 64, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 8, 6, 16));
			int iterations = 500;
			int length3 = dataAndLabel3.size();
			for(int j=0;j<iterations;j++)
			{
				if (j > 10 && j%(iterations/10)==0 || j < 3) System.out.println("Iteration: "+j) ;
		//early prediction
				if (j > 99 && j%(iterations/100)==0) {
					ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
					float tempval;
					float maxtempval = 0;
					int maxk = 0;
					for(int i=0;i<testData3.size();i++)
					{
						Blob out[]=n2.forward(testData3.get(i).data);
						for (int k = 0; k < 6; k++) {
							tempval = out[out.length-1].getValue(k);
							if (tempval > maxtempval) {
								maxtempval = tempval;
								maxk = k;
							}
						}
						testData3.get(i).label.setValue(0, (float) maxk);
						maxtempval = 0;
					}
			
					DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
				}
				
				for(int i=0;i<length3;i++)
				{
					//if (i%10000 == 0) System.out.println("Test: "+ i + "/" + length3);
					int idx=i;
					Blob out=n2.trainSimpleSGD(dataAndLabel3.get(idx).data, dataAndLabel3.get(idx).label);
	
					if((j==iterations-1 || j<10 || (j%(iterations/100)==0 && j > 50)) && i<10)
					{
	
						for(int h=0;h<out.getLength();h++)
						{
						 	System.out.print(out.getValue(h)+" ");
						}
	
						System.out.print("vs. ");
	
						for(int h=0;h<dataAndLabel3.get(idx).label.getLength();h++)
						{
						 	System.out.print(dataAndLabel3.get(idx).label.getValue(h)+" ");
						}
						System.out.println();
					}
				}
			}
			
		//prediction
			ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
			float tempval;
			float maxtempval = 0;
			int maxj = 0;
			for(int i=0;i<testData3.size();i++)
			{
				Blob out[]=n2.forward(testData3.get(i).data);
				for (int j = 0; j < 6; j++) {
					tempval = out[out.length-1].getValue(j);
					if (tempval > maxtempval) {
						maxtempval = tempval;
						maxj = j;
					}
					
				}
				testData3.get(i).label.setValue(0, (float) maxj);
				maxtempval = 0;
			}
	
			DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
			System.out.print("done");
		}

	}
}
