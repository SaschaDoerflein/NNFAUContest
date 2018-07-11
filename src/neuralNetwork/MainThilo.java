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
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 32*32 /*here*/, 16)); 
			//X Sigmoid Activation Function ist eine Alternative, wie viele FullyConnected Layer wir brauchen und wie groß sie sind
			//ist die große Frage!!!
			
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 64, 16));
			//n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 128, 64, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32 /*here*/, 32*16 /*here*/, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*16, 32, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32, 16, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 8/*here*/, 6, 16));
			
			//X 6 Outputs muss so bleiben// executionPrevent also der letzte Parameter (16) könnte auch noch verändert werden...
			
			int iterations = 100; //X 1/learning rate scheint eine gute Daumenregeln zu sein
			int length3 = dataAndLabel3.size();
			
			SimpleAccuracyFunction af = new SimpleAccuracyFunction();
			
			for(int j=1;j<iterations+1;j++)
			{
				ArrayList<Float> realError = new ArrayList<Float>();
				
				for(int i=0;i<length3;i++)
				{
					ArrayList<Float> result = new ArrayList<Float>();
					ArrayList<Float> expected = new ArrayList<Float>();

					int idx=i;
					Blob out=n2.trainSimpleSGD(dataAndLabel3.get(idx).data, dataAndLabel3.get(idx).label);
	
					if(j % 10 == 0)
					{
	
						for(int h=0;h<out.getLength();h++)
						{
						 	//System.out.print(out.getValue(h)+" ");
						 	result.add(out.getValue(h));
						}
	
						//System.out.print("vs. ");
	
						for(int h=0;h<dataAndLabel3.get(idx).label.getLength();h++)
						{
						 	//System.out.print(dataAndLabel3.get(idx).label.getValue(h)+" ");
						 	expected.add(dataAndLabel3.get(idx).label.getValue(h));
						}
						//System.out.println();
						
						af.setExpected(expected);
						af.setResult(result);
						
						realError.add(af.computeRealError());
					}
				}
				
				if(j % 10 == 0)
				{
					float sum = 0f;
					for (Float f : realError)
					{
						sum += f;
					}
					
					System.out.println(j + "] real error: " + sum / realError.size());
				}
			}
			
			//prediction
//			ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
//			float tempval;
//			float maxtempval = 0;
//			int maxj = 0;
//			for(int i=0;i<testData3.size();i++)
//			{
//				Blob out[]=n2.forward(testData3.get(i).data);
//				for (int j = 0; j < 6; j++) {
//					tempval = out[out.length-1].getValue(j);
//					if (tempval > maxtempval) {
//						maxtempval = tempval;
//						maxj = j;
//					}
//					
//				}
//				testData3.get(i).label.setValue(0, (float) maxj);
//				maxtempval = 0;
//			}
//	
//			DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
//			System.out.print("done");
	}
}