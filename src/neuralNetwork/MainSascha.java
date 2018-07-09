package neuralNetwork;
import java.io.IOException;
import java.util.*;

import test.SimpleAccuracyFunction;
import test.TestHelper;

import java.io.*;

public class MainSascha
{
	public static void main(String[] args) throws IOException
	{
		/*
		 * 
		 */
		
		
		
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
				/* minOutputCountLeft, maxOutputCountLeft,*/ minHiddenLayers, maxHiddenLayers, minHiddenCount, maxHiddenCount, 
				lfList, wfList, bfList,
				minExcecutionPrevent, maxExcecutionPrevent, excecutionPreventStepSize);
				
		
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
