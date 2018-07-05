package neuralNetwork;
import java.io.IOException;
import java.util.*;
import java.io.*;

public class MainChris
{
	public static void main(String[] args) throws IOException
	{
		//Testcases.doTestcases();
		int[] test = {0,1,0};

		/*---------------------------------------------------
		------ Example NeuralNet using Titanic-Dataset ------
		---------------------------------------------------*/
		if (test[0] == 1) {
			ArrayList<Datum> dataAndLabel=DataReader.readTitanicDataset("titanic_training.txt",true);
			int iterations = 1000000;
			int length = dataAndLabel.size();
			Network n=new Network(new VariantLearningRate(0.00005f,iterations,1,null));
			//Network n=new Network(new ConstantLearningRate(0.001f));
	
			n.add(new InputLayer(6));
			n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 4, 16));
			/*for (int i = 0; i < 20; i++) {
				n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 4, 4));
			}*/
			
			n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 4, 2, 16));
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
		}
		
		/*---------------------------------------------------
		------ Example NeuralNet using TravelTimeDataset ------
		---------------------------------------------------*/
		if (test[1] == 1) {
			ArrayList<Datum> dataAndLabel2=DataReader.readTraveltimeDataset("rta_training.txt",true);
			
			int iterations2 = 100;
			Network n2=new Network(new ConstantLearningRate(0.0007f));
			//Network n2=new Network(new VariantLearningRate(0.0001f,iterations2,1,null));
	
			n2.add(new InputLayerInverse(26));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 26, 16, 50));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 50));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 50));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 50));
			n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 16, 16, 50));
			n2.add(new OutputLayerInverse(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 16, 6, 50));
	
			for(int j=0;j<iterations2;j++)
			{
				if ((j > 10 && j%(iterations2/10)==0) || j < 3) System.out.println("Iteration: "+j) ;
	
				for(int i=0;i<dataAndLabel2.size();i++)
				{
					int idx=i;
					Blob out=n2.trainSimpleSGD(dataAndLabel2.get(idx).data, dataAndLabel2.get(idx).label);
	
					if((j==iterations2-1 || j<3 || ( j > 500 && j%1000==0)) && i<10)
					{
	
						for(int h=0;h<out.getLength();h++)
						{
						 	System.out.print(out.getValue(h)+" ");
						}
	
						System.out.print("vs. ");
	
						for(int h=0;h<dataAndLabel2.get(idx).label.getLength();h++)
						{
						 	System.out.print(dataAndLabel2.get(idx).label.getValue(h)+" ");
						}
						System.out.println();
					}
				}
			}
	
			ArrayList<Datum> testData2=DataReader.readTraveltimeDataset("rta_test1.txt",false);
			
			float tempval;
			for(int i=0;i<testData2.size();i++)
			{
				Blob out[]=n2.forward(testData2.get(i).data);
				for (int j = 0; j < out[out.length-1].getLength();j++) {
					tempval = Math.round(out[out.length-1].getValue(j));
					testData2.get(i).label.setValue(j,tempval);
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
	
			Network n2=new Network(new ConstantLearningRate(0.01f));
	
			n2.add(new InputLayer(32*32*3));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 256, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 256, 128, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 128, 64, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 32, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32, 16, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8, 16));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 8, 6, 16));
			int iterations = 1;
			int length3 = dataAndLabel3.size();
			for(int j=0;j<iterations;j++)
			{
				if (j > 10 && j%(iterations/10)==0 || j < 3) System.out.println("Iteration: "+j) ;
		//early prediction
				if (j > 100 && j%(iterations/100)==0) {
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
			}
	
			DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
			System.out.print("done");
		}

	}
}
