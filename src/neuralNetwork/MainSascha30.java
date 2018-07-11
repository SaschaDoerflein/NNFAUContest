package neuralNetwork;
import java.io.IOException;
import java.util.*;

import test.SimpleAccuracyFunction;
import test.TestHelper;

import java.io.*;

public class MainSascha30
{
	public static void main(String[] args) throws IOException
	{
		//Einstellungen für 30%
		
		ArrayList<Datum> dataAndLabel2=DataReader.getImageDataset("image_trainingbin.sec",true);
		ArrayList<Datum> dataAndLabel3 = DataReader.getCreyscale(dataAndLabel2,true);
		Network n2=new Network(new ConstantLearningRate(0.002f));

		n2.add(new InputLayer(32*32*1));
		n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 32*32*1, 128, 16));
		//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 64, 16));
		n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 128, 64, 16));
		//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 64, 32, 16));
		n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 64, 32, 16));
		n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 32, 16, 16));
		//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
		//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
		n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 16, 6, 16));
		int iterations = 100;
		int length3 = dataAndLabel3.size();
		
		
		for(int j=0;j<iterations;j++)
		{
			/*
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
			*/
			System.out.println("Iteration: "+j);
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
		ArrayList<Datum> testData2=DataReader.getImageDataset("image_testsetbin.sec",false);
		ArrayList<Datum> testData3 = DataReader.getCreyscale(testData2,false);
		
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
	

		
		
		
		
		
		
		
		
		
		
		
		/*
		 * 
	
		
		
		
		SimpleAccuracyFunction saf = new SimpleAccuracyFunction();
		TestHelper th = new TestHelper(saf);
		
		//Define Variables for brute force
		String nameOfTrainingsData = "titanic_training.txt";
		
		float minLearningRate = 0.00005f, maxLearningRate = 0.0001f, learningRateStepSize = 0.000005f;
		
		float minStartRate=1, maxStartRate=2, startRateStepSize=1f;
		
		int minIterations=1000, maxIterations=1001, iterationsStepSize=1;
		
		float minFactor=1, maxFactor=2, factorStepSize=1;
		
		ArrayList<ActivationFunction> activationFunctions = new ArrayList<>();
		//activationFunctions.add(new LinearActivation());
		activationFunctions.add(new SigmoidActivation());
		activationFunctions.add(new TanhActivation());
		
		int minExcecutionPrevent = 16, maxExcecutionPrevent=17, excecutionPreventStepSize=1;
		
		int inputCount=6, outputCountRight=1;
		int minHiddenLayers=2, maxHiddenLayers=4;
		int minHiddenCount=2, maxHiddenCount=16;
		
		ArrayList<LossFunction> lfList = new ArrayList<>();
		lfList.add(new EuclideanLoss());
		
		ArrayList<WeightFiller> wfList = new ArrayList<>();
		wfList.add(new RandomWeight());
		
		ArrayList<BiasFiller> bfList = new ArrayList<>();
		bfList.add(new ConstantBias());
		
		
		th.bruteForceTrainingsRun(nameOfTrainingsData, 
				minLearningRate, maxLearningRate, learningRateStepSize, 
				minStartRate, maxStartRate, startRateStepSize, 
				minIterations, maxIterations, iterationsStepSize, 
				minFactor, maxFactor, factorStepSize, activationFunctions, 
				inputCount, outputCountRight, 
				/* minOutputCountLeft, maxOutputCountLeft,*//* minHiddenLayers, maxHiddenLayers, minHiddenCount, maxHiddenCount, 
				lfList, wfList, bfList,
				minExcecutionPrevent, maxExcecutionPrevent, excecutionPreventStepSize);
				 */			
		
	/*
		SimpleAccuracyFunction saf = new SimpleAccuracyFunction();
			String nameOfTrainingsData = "rta_training.txt";
			
			float minLearningRate = 0.05f, maxLearningRate = 0.1f, learningRateStepSize = 0.05f;
			
			float minStartRate=1, maxStartRate=2, startRateStepSize=1f;
			
			int minIterations=1000, maxIterations=1001, iterationsStepSize=1;
			
			float minFactor=1, maxFactor=2, factorStepSize=1;
			
			ArrayList<LossFunction> lfList = new ArrayList<>();
			lfList.add(new EuclideanLoss());
			
			ArrayList<WeightFiller> wfList = new ArrayList<>();
			wfList.add(new RandomWeight());
			
			ArrayList<BiasFiller> bfList = new ArrayList<>();
			bfList.add(new ConstantBias());
			
			ArrayList<ActivationFunction> activationFunctions = new ArrayList<>();
			activationFunctions.add(new LinearActivation());
			activationFunctions.add(new SigmoidActivation());
			activationFunctions.add(new TanhActivation());
			
			int minExcecutionPrevent = 16, maxExcecutionPrevent=17, excecutionPreventStepSize=1;
			
			int inputCount=26, outputCountRight=6;
			int minHiddenLayers=2, maxHiddenLayers=10;
			int minHiddenCount=1, maxHiddenCount=32;
			
			TestHelper th = new TestHelper(saf);
			th.bruteForceTrainingsRun(nameOfTrainingsData, minLearningRate, maxLearningRate, learningRateStepSize,
					minStartRate, maxStartRate, startRateStepSize, minIterations, maxIterations, iterationsStepSize,
					minFactor, maxFactor, factorStepSize, activationFunctions, inputCount, outputCountRight, minHiddenLayers,
					maxHiddenLayers, minHiddenCount, maxHiddenCount, lfList, wfList, bfList, minExcecutionPrevent, maxExcecutionPrevent, 
					excecutionPreventStepSize);
			
		
*/
	}
	
}
