package neuralNetwork;

import java.io.IOException;
import java.util.ArrayList;

import test.SimpleAccuracyFunction;
import test.TestHelper;

public class MainThilo {
	public static void main(String[] args) throws IOException
	{
		// X heißt dort kann man gut was verändern und drann rumspielen!
		// Um mehr einsichten zu finden schau dir in Test SimpleAccuracyFunction an. Dort kannst du die kumulierte Abweichung von den 
		//Ergebnissen des Trainingsdatensets ausrechnen. Dann siehst du immer, ob du dich wenigstens annäherst.
		//Wie man sie anwendet kannst du in TestHelper ansehen. Dort werden zwei listen "expected" und "result" befüllt und dann ausgewertet.
		/*---------------------------------------------------
		------ Example NeuralNet using ImageDataset ------
		---------------------------------------------------*/
		
			
			ArrayList<Datum> dataAndLabel3=DataReader.getImageDataset("image_training.bin",true);
	
			Network n2=new Network(new ConstantLearningRate(0.001f));//X eher niedrige learning rates (<0.01) scheinen zu funktionieren
	
			n2.add(new InputLayer(32*32*3));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 16, 16)); 
			//X Sigmoid Activation Function ist eine Alternative, wie viele FullyConnected Layer wir brauchen und wie groß sie sind
			//ist die große Frage!!!
			
			
			
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 64, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 128, 64, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 8, 6, 16));
			
			//X 6 Outputs muss so bleiben// executionPrevent also der letzte Parameter (16) könnte auch noch verändert werden...
			
			int iterations = 500; //X 1/learning rate scheint eine gute Daumenregeln zu sein
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
		

	
		
		
		
		
		
		/*
		 * Hier ist der bereich für den Brute Force Algorithmus. Bei diesem dataset nicht zu empfehlen weil es zu lange dauert!!!
		 */
		
		/*
		SimpleAccuracyFunction saf = new SimpleAccuracyFunction();
		TestHelper th = new TestHelper(saf);
		
		//Define Variables for brute force
		String nameOfTrainingsData = "image_training.bin";
		
		float minLearningRate = 0.0005f, maxLearningRate = 0.001f, learningRateStepSize = 0.0005f;
		
		float minStartRate=1, maxStartRate=2, startRateStepSize=1f;
		
		int minIterations=500, maxIterations=501, iterationsStepSize=1;
		
		float minFactor=1, maxFactor=2, factorStepSize=1;
		
		ArrayList<ActivationFunction> activationFunctions = new ArrayList<>();
		//activationFunctions.add(new LinearActivation());
		activationFunctions.add(new SigmoidActivation()); //This is the best
		//activationFunctions.add(new TanhActivation());
		
		int minExcecutionPrevent = 16, maxExcecutionPrevent=17, excecutionPreventStepSize=1;
		
		int inputCount=32*32*3, outputCountRight=6; //In and out is fix!
		int minHiddenLayers=2, maxHiddenLayers=6;
		int minHiddenCount=8, maxHiddenCount=32*32*3;
		
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
				/* minOutputCountLeft, maxOutputCountLeft,*/ /*minHiddenLayers, maxHiddenLayers, minHiddenCount, maxHiddenCount, 
				lfList, wfList, bfList,
				minExcecutionPrevent, maxExcecutionPrevent, excecutionPreventStepSize);

				*/
	}
}
