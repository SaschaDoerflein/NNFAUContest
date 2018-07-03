package neuralNetwork;
import java.io.IOException;
import java.util.*;

import test.SimpleAccuracyFunction;
import test.TestHelper;

import java.io.*;

public class Main
{
	public static void main(String[] args) throws IOException
	{
		
		int[] test = {1,1,0};
		
		
		if (test[0]==1) {
			int iterations = 1000;
			int epochs = 1;
			float learningRate = 0.0001f;
			
			TestHelper th = new TestHelper(new SimpleAccuracyFunction());
			
			//Here you can create more networks to see if there is a varianz in the output networks. 
			Network[] networks = new Network[epochs];
			
			for(int i=0; i<epochs; i++) {
				Network n=new Network(new VariantLearningRate(learningRate,iterations));
				//Network n=new Network(new ConstantLearningRate(0.0001f));

				n.add(new InputLayer(6));
				
				for(int ii = 0; ii<0;ii++) {
					n.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 6));
				}
				
				n.add(new OutputLayer(new EuclideanLoss(),new LinearActivation(), new RandomWeight(), new ConstantBias(), 6, 1));
				networks[i] = n;
			}
			
			
			th.performTest(networks, iterations,"titanic_training.txt");
			
			/* TODO: prediction make no sense right now because we have no expected value
			 * 
			 * ArrayList<Datum> testData=DataReader.readTitanicDataset("titanic_test1.txt",false);
	
			for(int i=0;i<testData.size();i++)
			{
				Blob out[]=n.forward(testData.get(i).data);
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
			System.out.println("done");
			 */

		}
		
		if (test[1] == 1) {		
			int iterations = 10000;
			int epochs = 1;
			float learningRate = 0.0000000007f;
			
			TestHelper th = new TestHelper(new SimpleAccuracyFunction());
			
			//Here you can create more networks to see if there is a varianz in the output networks. 
			Network[] networks = new Network[epochs];
			
			for(int i=0; i<epochs; i++) {
				Network n=new Network(new VariantLearningRate(learningRate,iterations));
				//Network n=new Network(new ConstantLearningRate(0.0001f));

				n.add(new InputLayer(6));
				
				for(int ii = 0; ii<0;ii++) {
					n.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 6));
				}
				
				n.add(new OutputLayer(new EuclideanLoss(),new LinearActivation(), new RandomWeight(), new ConstantBias(), 6, 1));
				networks[i] = n;
			}
			th.performTest(networks, iterations,"rta_training.txt");
	/*
			ArrayList<Datum> testData2=DataReader.readTraveltimeDataset("rta_test1.txt",false);
	
			for(int i=0;i<testData2.size();i++)
			{
				Blob out[]=n2.forward(testData2.get(i).data);
				if(out[out.length-1].getValue(0) < 0.5)
				{
					testData2.get(i).label.setValue(0,0f);
				}
				else
				{
					testData2.get(i).label.setValue(0,1f);
				}
			}

			DataWriter.writeLabelsToFile("rta_prediction.txt", testData2);
			*/
		}
		
		
		
		/*---------------------------------------------------
		------ Example NeuralNet using ImageDataset ------
		---------------------------------------------------
		if (test[2] == 1) {
			
			ArrayList<Datum> dataAndLabel3=DataReader.getImageDataset("image_training.bin",true);
	
			Network n2=new Network(new ConstantLearningRate(0.0000001f));
	
			n2.add(new InputLayer(32*32*3));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 200));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 1));
			int iterations = 1000;
			int length3 = dataAndLabel3.size();
			for(int j=0;j<iterations;j++)
			{
			   System.out.println("Iteration: "+j);
			   	
				for(int i=0;i<length3;i++)
				{
					//if (i%3000 == 0) System.out.println("Test: "+ i + "/" + length3);
					int idx=i;
					Blob out=n2.trainSimpleSGD(dataAndLabel3.get(idx).data, dataAndLabel3.get(idx).label);
	
					if((j==iterations-1 || j<10) && i<100)
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
	
			ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
	
			for(int i=0;i<testData3.size();i++)
			{
				Blob out[]=n2.forward(testData3.get(i).data);
				testData3.get(i).label.setValue(0, Math.round(out[out.length-1].getValue(0)));
				/*if(out[out.length-1].getValue(0) < 0.5)
				{
					testData3.get(i).label.setValue(0,0f);
				}
				else
				{
					testData3.get(i).label.setValue(0,1f);
				}
			}
	
			DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
			
		}
*/
	}
}
