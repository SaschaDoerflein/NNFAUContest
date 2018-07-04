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
		TestHelper th = new TestHelper(new SimpleAccuracyFunction());
		
		int[] test = {1,1,0};
		
		
		if (test[0]==1) {
			int iterations = 1000;
			int epochs = 1;
			float learningRate = 0.0001f;
			
			//Here you can create more networks to see if there is a varianz in the output networks. 
			Network[] networks = new Network[epochs];
			
			for(int i=0; i<epochs; i++) {
				Network n=new Network(new VariantLearningRate(learningRate,iterations));
				//Network n=new Network(new ConstantLearningRate(0.0001f));

				n.add(new InputLayer(6));
				
				//In every epoch there are more FullyConnected Layers
				for(int ii = 0; ii<i;ii++) {
					n.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 6));
				}
				
				n.add(new OutputLayer(new EuclideanLoss(),new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 1));
				networks[i] = n;
			}
			
			
			ArrayList<Datum> dataAndLabel = dataAndLabel = DataReader.readTitanicDataset("titanic_training.txt",true);

			th.performTest(networks, iterations,dataAndLabel);
			
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
			
		
			
			//Here you can create more networks to see if there is a varianz in the output networks. 
			Network[] networks = new Network[epochs];
			
			for(int i=0; i<epochs; i++) {
				Network n=new Network(new VariantLearningRate(learningRate,iterations));
				//Network n=new Network(new ConstantLearningRate(0.0001f));

				n.add(new InputLayer(26));
				
				for(int ii = 0; ii<0;ii++) {
					n.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 6));
				}
				
				n.add(new OutputLayer(new EuclideanLoss(),new LinearActivation(), new RandomWeight(), new ConstantBias(), 26, 6));
				networks[i] = n;
			}
			
		
			ArrayList<Datum> dataAndLabel = DataReader.readTraveltimeDataset("rta_training.txt",true);
		
		
			
			th.performTest(networks, iterations,dataAndLabel);
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
		--------------------------------------------------- */
		
		if (test[2] == 1) {		
			int iterations = 1000;
			int epochs = 1;
			float learningRate = 0.0000001f;
			
			//Here you can create more networks to see if there is a varianz in the output networks. 
			Network[] networks = new Network[epochs];
			
			for(int i=0; i<epochs; i++) {
				Network n=new Network(new ConstantLearningRate(learningRate));
				//Network n=new Network(new ConstantLearningRate(0.0001f));

				n.add(new InputLayer(32*32*3));
				
				for(int ii = 0; ii<0;ii++) {
					n.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 6, 6));
				}
				
				n.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 1));
				networks[i] = n;
			}
			
			ArrayList<Datum> dataAndLabel = DataReader.getImageDataset("image_training.bin",true);
			
			th.performTest(networks,iterations,dataAndLabel);
		
			/*
			 * ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
	
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
			 */
			
		}
	}
	
	
	private Network[] BuildNetworks(int numInputLayers, int minMiddleLayers, int maxMiddleLAyers, int numOuputLayers) {
		//Take all Learning rates
		Network n1 =new Network(new VariantLearningRate(learningRate,iterations));
		Network n2 = new Network(new ConstantLearningRate(learningRate));
		
		
		
		
		
	}
	
	
	
	
	/*
	 * In this section there tests to find the optimum over night ;)
	 */
	
	
	//Probiere alle möglichkeiten von min nach max aus (kann leicht für weitere Functionen erweitert werden.
	private void intoTheBlue(int maxLayers, int minLayers, ArrayList<ActivationFunction> afList, ArrayList<LearningRate> lrList, int maxIterations, int minIterations, int iterationStepSize) {
		
	}
	
	//Try to get best learningRate between max and min
	private void intoTheBlueLearningRate(float max, float min) {
		
	}
	
	
}
