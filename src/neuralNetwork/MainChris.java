package neuralNetwork;
import java.io.IOException;
import java.util.*;
import java.io.*;

public class MainChris
{
	public static void main(String[] args) throws IOException
	{
		//Testcases.doTestcases();
		int[] test = {1,0,0};

		/*---------------------------------------------------
		------ Example NeuralNet using Titanic-Dataset ------
		---------------------------------------------------*/
		if (test[0] == 1) {
			ArrayList<Datum> dataAndLabel=DataReader.readTitanicDataset("titanic_training.txt",true);
			int iterations = 50001;
			int length = dataAndLabel.size();
			Network n=new Network(new VariantLearningRate(0.0001f,iterations,1,null));
			//Network n=new Network(new ConstantLearningRate(0.001f));
	
			n.add(new InputLayer(6));
			//n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 6, 4));
			/*for (int i = 0; i < 20; i++) {
				n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 4, 4));
			}*/
			
			//n.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 600, 60));
			n.add(new OutputLayer(new EuclideanLoss(),new LinearActivation(), new RandomWeight(), new ConstantBias(), 6, 1));
			
			
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
			
			int iterations2 = 10000;
			Network n2=new Network(new ConstantLearningRate(0.0000000007f));
			//Network n2=new Network(new VariantLearningRate(0.0001f,iterations2,1,null));
	
			n2.add(new InputLayer(26));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 26, 40));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 26, 6));
	
			for(int j=0;j<iterations2;j++)
			{
			   System.out.println("Iteration: "+j);
	
				for(int i=0;i<dataAndLabel2.size();i++)
				{
					int idx=i;
					Blob out=n2.trainSimpleSGD(dataAndLabel2.get(idx).data, dataAndLabel2.get(idx).label);
	
					if((j==iterations2-1 || j<2) && i<10)
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
		}
		
		/*---------------------------------------------------
		------ Example NeuralNet using ImageDataset ------
		---------------------------------------------------*/
		if (test[2] == 1) {
			
			ArrayList<Datum> dataAndLabel3=DataReader.getImageDataset("image_training.bin",true);
	
			Network n2=new Network(new ConstantLearningRate(0.01f));
	
			n2.add(new InputLayer(32*32*3));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32*32*3, 256));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 256, 128));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 128, 64));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 64, 32));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 32, 16));
			n2.add(new FullyConnected(new TanhActivation(), new RandomWeight(), new ConstantBias(), 16, 8));
			//n2.add(new FullyConnected(new SigmoidActivation(), new RandomWeight(), new ConstantBias(), 40, 10));
			n2.add(new OutputLayer(new EuclideanLoss(), new LinearActivation(), new RandomWeight(), new ConstantBias(), 8, 1));
			int iterations = 1000;
			int length3 = dataAndLabel3.size();
			for(int j=0;j<iterations;j++)
			{
				if (j > 10 && j%(iterations/10)==0 || j < 3) System.out.println("Iteration: "+j) ;
				if (j > 100 && j%(iterations/100)==0) {
					ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
					float tempval;
					for(int i=0;i<testData3.size();i++)
					{
						Blob out[]=n2.forward(testData3.get(i).data);
						tempval = out[out.length-1].getValue(0);
						if (tempval < 0) tempval = 0;
						if (tempval > 5) tempval = 5;
						testData3.get(i).label.setValue(0, Math.round(tempval));
					}					
		
				DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
				System.out.print(" some done");
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
	
			ArrayList<Datum> testData3=DataReader.getImageDataset("image_test1.bin",false);
			float tempval;
			for(int i=0;i<testData3.size();i++)
			{
				Blob out[]=n2.forward(testData3.get(i).data);
				tempval = out[out.length-1].getValue(0);
				if (tempval < 0) tempval = 0;
				if (tempval > 5) tempval = 5;
				testData3.get(i).label.setValue(0, Math.round(tempval));
				/*if(out[out.length-1].getValue(0) < 0.5)
				{
					testData3.get(i).label.setValue(0,0f);
				}
				else
				{
					testData3.get(i).label.setValue(0,1f);
				}*/
			}
	
			DataWriter.writeLabelsToFile("image_prediction.txt", testData3);
			System.out.print("done");
		}

	}
}
