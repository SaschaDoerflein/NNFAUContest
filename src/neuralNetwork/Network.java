package neuralNetwork;
import java.util.ArrayList;

/// ----------------------------------------------------------------------------------------
/// This class represents the whole network. It also offers functions to train layers
/// and propagate forward a given input.
/// ----------------------------------------------------------------------------------------

public class Network
{
	ArrayList<Layer> layers;
	LearningRate learningRate;

   public Network(LearningRate learningRate)
   {
	   this.learningRate=learningRate;
	   layers=new ArrayList<Layer>();
   }

   // Adds one layer to network
   public void add(Layer l)
   {
	   layers.add(l);
   }

   // Forwards an input through the network
   public Blob[] forward(Blob in)
   {
	   Blob[] forward=new Blob[layers.size()+1];
	   forward[0]=in;
	   for(int i=0;i<layers.size();i++)
	   {
		   forward[i+1]=layers.get(i).forward(forward[i]);
	   }
	   return forward;
   }

   // Trains the network using basic stochastic gradient descent (SGD)
   // Returns the output of the last layer, which can be used to calculate loss.
   public Blob trainSimpleSGD(Blob in, Blob out)
   {
	   Blob[] forward=forward(in);

	   Blob backward=out;
	   for(int i=layers.size()-1;i>=0;i--)
	   {
		   if(i==layers.size()-1)
			   backward=layers.get(i).backward(backward,null);
		   else
			   backward=layers.get(i).backward(backward,layers.get(i+1).getWeights());
	   }

	   float currRate=learningRate.getLearningRate();
	   for(int i=0;i<layers.size();i++)
	   {
		   if(i==0)
			   layers.get(i).updateWeightsAndBias(in, currRate);
		   else
			   layers.get(i).updateWeightsAndBias(forward[i], currRate);
	   }

	   return forward[forward.length-1];
   }
   
   public Blob forwardLSTM(Blob in) {
	   
	   Blob[] forward0=new Blob[in.getLength()+6];
	   Blob[] forward=new Blob[in.getLength()+6];
	   Blob[] forward2=new Blob[in.getLength()+6];
	   Blob forward3 = new Blob(6);
	   //forward[0].set=in.getValue(0);
	   float div = 3000;
	   for(int i=0;i<in.getLength();i++)
	   {
		   forward[i] = new Blob(1);
		   forward[i].setValue(0, in.getValue(i)/div);
		   forward0[i] = new Blob(1);
		   forward0[i].setValue(0, in.getValue(i)/div);
	   }
	   for(int i=0;i<6;i++)
	   {
		   forward[i+in.getLength()] = new Blob(1);
		   //forward[i+in.getLength()].setValue(0, out.getValue(i)/div);
		   forward0[i+in.getLength()] = new Blob(1);
		   //forward0[i+in.getLength()].setValue(0, out.getValue(i)/div);
	   }
	   float currRate=learningRate.getLearningRate();
	   for(int i=0;i<in.getLength()+6-1;i++)
	   {
		   forward2[i+1] = layers.get(0).forward(forward[i]);
		   layers.get(0).backward(forward0[i+1], null);
		   layers.get(0).updateWeightsAndBias(forward[i], currRate);
		   forward0[i+1].setValue(0, forward2[i+1].getValue(0));
	   }
	   /*for (int i=in.getLength()-1;i<in.getLength()+6-1;i++) {
		   forward2[i+1] = layers.get(0).forward(forward[i]);
		   layers.get(0).backward(forward0[i+1], null);
		   layers.get(0).updateWeightsAndBias(forward[i], currRate);
		   forward[i+1].setValue(0, forward2[i+1].getValue(0));
		   forward[i+1].setValue(1, forward2[i+1].getValue(0));
	   }*/
	   for (int i = 0; i < 6; i++) {
		   forward3.setValue(i, forward0[i+in.getLength()].getValue(0)*div);
	   }
	   return forward3;
   }
   public Blob trainLSTM(Blob in, Blob out) {
	   
	   Blob[] forward0=new Blob[in.getLength()+out.getLength()];
	   Blob[] forward=new Blob[in.getLength()+out.getLength()];
	   Blob[] forward2=new Blob[in.getLength()+out.getLength()];
	   Blob forward3 = new Blob(out.getLength());
	   //forward[0].set=in.getValue(0);
	   float div = 3000;
	   for(int i=0;i<in.getLength();i++)
	   {
		   forward[i] = new Blob(1);
		   forward[i].setValue(0, in.getValue(i)/div);
		   forward0[i] = new Blob(1);
		   forward0[i].setValue(0, in.getValue(i)/div);
	   }
	   for(int i=0;i<out.getLength();i++)
	   {
		   forward[i+in.getLength()] = new Blob(1);
		   forward[i+in.getLength()].setValue(0, out.getValue(i)/div);
		   forward0[i+in.getLength()] = new Blob(1);
		   forward0[i+in.getLength()].setValue(0, out.getValue(i)/div);
	   }
	   float currRate=learningRate.getLearningRate();
	   for(int i=0;i<in.getLength()+out.getLength()-1;i++)
	   {
		   forward2[i+1] = layers.get(0).forward(forward[i]);
		   layers.get(0).backward(forward0[i+1], null);
		   layers.get(0).updateWeightsAndBias(forward[i], currRate);
		   forward0[i+1].setValue(0, forward2[i+1].getValue(0));
	   }
	   for (int i = 0; i < out.getLength(); i++) {
		   forward3.setValue(i, forward0[i+in.getLength()].getValue(0)*div);
	   }
	   return forward3;
   }


    public Blob trainMinibatchSGD(Blob in, Blob out)
   {
	   Blob[] forward=forward(in);

	   Blob backward=out;
	   for(int i=layers.size()-1;i>=0;i--)
	   {
		   if(i==layers.size()-1)
			   backward=layers.get(i).backward(backward,null);
		   else
			   backward=layers.get(i).backward(backward,layers.get(i+1).getWeights());
	   }


		   float currRate=learningRate.getLearningRate();
		   for(int i=0;i<layers.size();i++)
		   {
			   if(i==0)
				   layers.get(i).updateWeightsAndBias(in, currRate);
			   else
				   layers.get(i).updateWeightsAndBias(forward[i], currRate);
		   }


	   return forward[forward.length-1];
   }

}