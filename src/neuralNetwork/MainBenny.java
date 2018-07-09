package neuralNetwork;

import java.io.IOException;
import java.util.ArrayList;

import test.SimpleAccuracyFunction;
import test.TestHelper;

public class MainBenny {
	public static void main(String[] args) throws IOException
	{
		SimpleAccuracyFunction saf = new SimpleAccuracyFunction();
		TestHelper th = new TestHelper(saf);
		
		//Define Variables for brute force
		String nameOfTrainingsData = "image_training.bin";
		
		float minLearningRate = 0.00005f, maxLearningRate = 0.001f, learningRateStepSize = 0.00005f;
		
		float minStartRate=1, maxStartRate=2, startRateStepSize=1f;
		
		int minIterations=100, maxIterations=101, iterationsStepSize=1;
		
		float minFactor=1, maxFactor=2, factorStepSize=1;
		
		ArrayList<ActivationFunction> activationFunctions = new ArrayList<>();
		//activationFunctions.add(new LinearActivation());
		activationFunctions.add(new SigmoidActivation()); //This is the best
		//activationFunctions.add(new TanhActivation());
		
		int minExcecutionPrevent = 16, maxExcecutionPrevent=17, excecutionPreventStepSize=1;
		
		int inputCount=32*32*3, outputCountRight=6; //In and out is fix!
		int minHiddenLayers=2, maxHiddenLayers=6;
		int minHiddenCount=2, maxHiddenCount=32*32*3;
		
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


	}
}
