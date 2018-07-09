package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;

import neuralNetwork.ActivationFunction;
import neuralNetwork.BiasFiller;
import neuralNetwork.Blob;
import neuralNetwork.ConstantBias;
import neuralNetwork.DataReader;
import neuralNetwork.Datum;
import neuralNetwork.EuclideanLoss;
import neuralNetwork.FullyConnected;
import neuralNetwork.InputLayer;
import neuralNetwork.LinearActivation;
import neuralNetwork.LossFunction;
import neuralNetwork.Network;
import neuralNetwork.OutputLayer;
import neuralNetwork.RandomWeight;
import neuralNetwork.SigmoidActivation;
import neuralNetwork.TanhActivation;
import neuralNetwork.VariantLearningRate;
import neuralNetwork.WeightFiller;

public class TestHelper {
	private int count;
	private AccuracyFunction af;
	
	public TestHelper(AccuracyFunction af) {
		this.af = af;
	}
	
	//Write every result from every iterations separated in epochs into a output file
	public void performTests(Network[] networks, int iterations, ArrayList<Datum> dataAndLabel) {
		count++;
		System.out.println("------------Starting Test"+count+"------------");
		float[] realErrors = new float[networks.length];
		float[] variances = new float[networks.length];
		int[][] errorClassesArray = new int[networks.length][];
			
		int length = dataAndLabel.size();	
			
			//Normalisation
			//TODO: Sascha 4.7.  Should be good but makes thinks worse :(
			/*
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
			*/
			
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
						 	result.add(out.getValue(h));//Sascha 7.4. if normalization is on: *maxes[h]
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
			
		
	}
	
	//Beware: (max-min)/StepSize must be even!
	public void bruteForceTrainingsRun(String nameOfTrainingsData,
			float minLearningRate, float maxLearningRate, float learningRateStepSize,
			float minStartRate, float maxStartRate, float startRateStepSize,
			int minIterations, int maxIterations, int iterationsStepSize, 
			float minFactor, float maxFactor,float factorStepSize,
			ArrayList<ActivationFunction> activationFunctions,
			int inputCount, int outputCountRight, 
			int minOutputCountLeft, int maxOutputCountLeft,
			int minHiddenLayers, int maxHiddenLayers,
			int minHiddenCount, int maxHiddenCount,
			ArrayList<LossFunction> lfList, ArrayList<WeightFiller> wfList, ArrayList<BiasFiller> bfList,
			int minExcecutionPrevent, int maxExcecutionPrevent, int excecutionPreventStepSize) {
		
		
		
		//**Variables**
		//You can compute how many loops the bruteForceTrainingRuns 
		//Beware allCombinations shouldn't be greater than 2 hoch Integer.MaxInteger()
		BigInteger learningRateCombinations =  BigInteger.valueOf((long) ((maxLearningRate - minLearningRate)/learningRateStepSize));
		BigInteger startrateCombinations =  BigInteger.valueOf((long) ((maxStartRate - minStartRate)/startRateStepSize));
		BigInteger iterationCombinations =  BigInteger.valueOf((long) ((maxIterations - minIterations)/iterationsStepSize));
		BigInteger factorCombinations =  BigInteger.valueOf((long)((maxFactor - minFactor)/factorStepSize));
		BigInteger outputCountLeftCombinations =  BigInteger.valueOf((long)(maxOutputCountLeft-minOutputCountLeft));
		BigInteger hiddenLayersCombinations =  BigInteger.valueOf((long)maxHiddenLayers-minHiddenLayers);
		BigInteger hiddenCountCombinations =  BigInteger.valueOf((long)maxHiddenCount*minHiddenCount);
		BigInteger activationFunctionsCombination = BigInteger.valueOf((long)activationFunctions.size());
		BigInteger excecutionPreventCombinations = BigInteger.valueOf((long)((maxExcecutionPrevent-minExcecutionPrevent)/excecutionPreventStepSize));
		BigInteger lfListCombinations = BigInteger.valueOf((long)lfList.size());
		BigInteger wfListCombinations = BigInteger.valueOf((long)wfList.size());
		BigInteger bfListCombinations = BigInteger.valueOf((long)bfList.size());
		
		
		BigInteger allCombinations = new BigInteger("1");
		allCombinations = allCombinations.multiply(learningRateCombinations);
		allCombinations = allCombinations.multiply(startrateCombinations);
		allCombinations = allCombinations.multiply(iterationCombinations);
		allCombinations = allCombinations.multiply(factorCombinations);
		allCombinations = allCombinations.multiply(outputCountLeftCombinations);
		allCombinations = allCombinations.multiply(hiddenLayersCombinations);
		allCombinations = allCombinations.multiply(hiddenCountCombinations);
		allCombinations = allCombinations.multiply(activationFunctionsCombination);
		allCombinations = allCombinations.multiply(excecutionPreventCombinations);
		allCombinations = allCombinations.multiply(lfListCombinations);
		allCombinations = allCombinations.multiply(wfListCombinations);
		allCombinations = allCombinations.multiply(bfListCombinations);

		
		BigInteger thisCombination = new BigInteger("1");
		final BigInteger oneBigInteger = new BigInteger("1");
		
		//Get Data
		try {
			ArrayList<Datum> dataAndLabel = DataReader.readTitanicDataset(nameOfTrainingsData,true);
			int length = dataAndLabel.size();
			
			//while(!allCombinations.equals(thisCombination)) {
				//System.out.println(thisCombination);
				
				for(float currentLearningRate = minLearningRate; currentLearningRate < maxLearningRate; currentLearningRate += learningRateStepSize) {
					for(float currentStartRate = minStartRate; currentStartRate < maxStartRate; currentStartRate += startRateStepSize) {
						for(int currentIteration = minIterations; currentIteration < maxIterations; currentIteration += iterationsStepSize) {
							for(float currentFactor = minFactor; currentFactor < maxFactor; currentFactor += factorStepSize) {
								for(int currentActivationFunction = 0; currentActivationFunction < activationFunctions.size(); currentActivationFunction++) {
									for(int currentExcecutionPrevent = minExcecutionPrevent; currentExcecutionPrevent< maxExcecutionPrevent; currentExcecutionPrevent += excecutionPreventStepSize) {
									//for(int currentOutputLeft = minOutputCountLeft; currentOutputLeft < maxOutputCountLeft; currentOutputLeft++) {
										for(int currentHiddenLayer = minHiddenLayers; currentHiddenLayer < maxHiddenLayers; currentHiddenLayer++) {
											for(int currentlfList = 0; currentlfList < lfList.size(); currentlfList++) {
												for(int currentwfList = 0; currentwfList < wfList.size(); currentwfList++) {
													for(int currentbfList = 0; currentbfList < bfList.size(); currentbfList++) {
														
														
															//Combinations for hidden layer...
														int[] hiddenConnections = null;

														if(minHiddenLayers > 0) {
															hiddenConnections = new int[currentHiddenLayer];
															for(int i = 0; i < hiddenConnections.length; i++) {
																hiddenConnections[i] = minHiddenCount;
															}
																int pivotElement = 0;
																
																while(hiddenConnections[hiddenConnections.length-1]< maxHiddenCount) {
																	while(hiddenConnections[pivotElement] < maxHiddenCount) {
																		
																		af = new SimpleAccuracyFunction();
																		
																		ArrayList<Float> result = af.getResult();
																		ArrayList<Float> expected = af.getExpected();
																		
																		//Set Objects
																		LossFunction lossFunction = lfList.get(currentlfList);
																		WeightFiller weightFiller = wfList.get(currentwfList);
																		BiasFiller biasFiller = bfList.get(currentbfList);
																		
																		ActivationFunction activationFunction = activationFunctions.get(currentActivationFunction);
																		VariantLearningRate variantLearningRate = new VariantLearningRate(currentStartRate,currentIteration,currentFactor,activationFunction);
																		
																		
																		//Build Network
																		Network network = new Network(variantLearningRate);

																		//TODO: When I add one layers there are 9 layers=null in the list :/
																		network.add(new InputLayer(inputCount));
																		
																		for(int hidden=1; hidden<=currentHiddenLayer+1; hidden++) {
																			//Just when in and out are same number
																			//You have #hidden-1 numbers of connections=#con    + input and output connection
																			//there are #con over currentHidden-minHidden combinations
																			if (hidden == 1) {
																				network.add(new FullyConnected(activationFunction,weightFiller,biasFiller,inputCount,hiddenConnections[hidden-1],currentExcecutionPrevent));
																			}else if (hidden == currentHiddenLayer+1) {
																				network.add(new OutputLayer(lossFunction,activationFunction,weightFiller,biasFiller,hiddenConnections[hidden-3],outputCountRight,currentExcecutionPrevent));
																			}else {
																				network.add(new FullyConnected(activationFunction,weightFiller,biasFiller,hiddenConnections[hidden-2],hiddenConnections[hidden-1],currentExcecutionPrevent));
																			}
																			
																		}
																		
																		//network.add(new OutputLayer(lossFunction,activationFunction, weightFiller, biasFiller, currentOutputLeft, outputCountRight, currentExcecutionPrevent));
																		
																		
																		//Perform Test
																		Blob out=null;
																		for(int iter=0 ; iter < currentIteration; iter++) {
																			for(int i = 0; i<length; i++) {
																				out=network.trainSimpleSGD(dataAndLabel.get(i).data, dataAndLabel.get(i).label);
																				if (iter == currentIteration-1) {
																					for(int h=0;h<out.getLength();h++)
																					{
																						result.add(out.getValue(h));
																					}
																
																					for(int h=0;h<dataAndLabel.get(i).label.getLength();h++)
																					{
																						expected.add(dataAndLabel.get(i).label.getValue(h));
																					}
																				}
																				
																			}
																		}
																		
																		
																		float realError = af.computeRealError();
																		float variance = af.computeVar();
																		int[] errorClasses = af.computeSimpleErrorClasses();
																		
																		
																		
																		//Store Result
																		StringBuilder sb = new StringBuilder();
																		sb.append(realError+"/"+variance+" ");
																		
																		for(int i=0;i<errorClasses.length;i++) {
																			sb.append(errorClasses[i]+" ");
																		}
																		
																		sb.append("HLayers: ");
																		
																		sb.append(currentHiddenLayer+" ");
																		
																		
																		
																		for(int i=0; i<hiddenConnections.length;i++) {
																			sb.append(hiddenConnections[i]+" ");
																		}
																		
																		sb.append(currentLearningRate+" ");
																		sb.append(currentStartRate+" ");
																		sb.append(currentIteration+" ");
																		sb.append(currentFactor+" ");
																		sb.append(currentActivationFunction+" ");
																		sb.append(currentExcecutionPrevent+" ");
																		//sb.append(currentOutputLeft+" ");
																		sb.append(currentlfList+" ");
																		sb.append(currentwfList+" ");
																		sb.append(currentbfList);//Don't forget the space
																		
																		System.out.println("Performed test "+sb.toString());
																		
																		try
																		{
																			//List of all results
																			File file = new File("results.txt");
																			//Just store the best variance result
																			File file2 = new File("result.txt");
																			// if file doesnt exists, then create it 
																			if ( ! file.exists( ) )
																			{
																				
																				file.createNewFile( );
																			}
																			if ( ! file2.exists( ) )
																			{
																				
																				file2.createNewFile( );
																			}

																			FileWriter fw = new FileWriter( file.getAbsoluteFile( ) ,true);
																			BufferedWriter bw = new BufferedWriter( fw );
																			bw.write( sb.toString());
																			bw.close( );
																			
																			FileReader fr = new FileReader(file2.getAbsolutePath());
																			BufferedReader br = new BufferedReader(fr);
																			 
																			String st;
																			st = br.readLine();
																			br.close();
																			if (st != null) {
																				System.out.println(sb);
																				st = st.substring(0, st.indexOf(" "));
																				float oldVariance = Float.parseFloat(st);
																				if(variance < oldVariance) {
																					FileWriter fw2 = new FileWriter(file2.getAbsolutePath());
																					BufferedWriter bw2 = new BufferedWriter( fw );
																					
																					bw2.write( sb.toString());
																					bw2.close( );
																					fw2.close();
																				}
																			}else {
																				FileWriter fw2 = new FileWriter(file2.getAbsolutePath());
																				BufferedWriter bw2 = new BufferedWriter( fw );
																				bw2.write( sb.toString());
																				
																			}
																		}
																		catch( IOException e )
																		{
																			System.out.println("Error: " + e);
																			e.printStackTrace( );
																		}
																		
																		//clean
																		af.setExpected(new ArrayList<Float>());
																		af.setResult(new ArrayList<Float>());

																		hiddenConnections[pivotElement]++;
																	}
																	pivotElement++;
																	if(hiddenConnections.length < pivotElement) {
																		for(int i=0; i<pivotElement;i++) {
																			hiddenConnections[pivotElement] = minHiddenCount;
																		}
																		pivotElement = 0;
																	}
																}
															}else {
																hiddenConnections = new int[0];
															}
																
																
																	
															}
																
														}
														
														
													}
												}
											}
										}
									//}
									}
								}
							}
						}
					}
	 catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
				}
	
}
