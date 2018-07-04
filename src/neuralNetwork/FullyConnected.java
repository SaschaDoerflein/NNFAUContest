package neuralNetwork;

/** class FullyConnected.
*/
public class FullyConnected implements Layer
{
	// Weights from all neurons of layer before to all neurons of current layer
	Blob weights;
	// Activation function for neurons
	ActivationFunction func;
	// The output of each neuron after applying activation functions etc.
	Blob output;
	// neuronDelta/gradient of each neuron
	Blob neuronDelta;
	// Bias for each neuron
	Blob bias;
	// You may need a temporary variable for partially processed output (if you do not need this variable, just ignore it)
	Blob tempOut;
	
	int execution;
	//Prevents updateWeight until exPre
	int executionPrevent;
	Blob tempW;
	Blob tempB;
   public Blob forward(Blob inputBlob)
   {
      for (int j=0;j<output.values.length;j++)
      {
        float sum=0f;
        for (int i=0;i<inputBlob.values.length;i++)
      	{
			sum+=inputBlob.getValue(i)*weights.getValue(i,j);
      	}
		tempOut.setValue(j,sum+bias.getValue(j));
        output.values[j]=func.compute(tempOut.getValue(j));
	}
	return output;
   }

   public Blob backward (Blob deltaBefore, Blob weightsBefore)
   {
      for (int j=0;j<neuronDelta.getLength();j++)
      {
    	  float error=0f;
    	  for (int i=0;i<deltaBefore.getLength();i++)
	      {
	     		error+= deltaBefore.getValue(i)*weightsBefore.getValue(j,i);
	      }
	      neuronDelta.setValue(j,error * func.derivative(tempOut.getValue(j)));
      }
      return neuronDelta;
   }

	public void updateWeightsAndBias(Blob inputBlob, float learningRate)
	{
		for (int j=0;j<neuronDelta.getLength();j++)
	      {

		      for (int i=0;i<inputBlob.getLength();i++)
		      {
		     		tempW.addValue(i,j,neuronDelta.getValue(j)*inputBlob.getValue(i));
		      }
		      tempB.addValue(j,neuronDelta.getValue(j));
	      }

	     	execution++;
	      if(execution==executionPrevent)
	      {

	      	for (int j=0;j<neuronDelta.getLength();j++)
		      {
			      for (int i=0;i<inputBlob.getLength();i++)
			      {
			     		weights.addValue(i,j,tempW.getValue(i,j)*learningRate/(float)execution) ;
			     		tempW.setValue(i,j,0f);
			      }
			      bias.addValue(j, tempB.getValue(j)*learningRate/(float)execution);
			      tempB.setValue(j,0f);
		      }
		      execution=0;
	      }
	}

	public FullyConnected(ActivationFunction func, WeightFiller fillerWeight,BiasFiller fillerBias , int in, int out, int executionPrevent)
	{
		this.executionPrevent = executionPrevent;
		execution=0;
	        		tempW=new Blob(in,out);
      	tempB=new Blob(out);
	   	tempOut=new Blob(out);
		output=new Blob(out);
		neuronDelta=new Blob(out);
		weights=new Blob(in,out);
		bias=new Blob(out);

		for(int i=0;i<bias.getLength();i++)
		{
			bias.setValue(i,fillerBias.compute(i, in, out));
			tempB.setValue(i,0f);
		}

		for(int i=0;i<in;i++)
		{
			for(int j=0;j<out;j++)
			{
				weights.setValue(i,j,fillerWeight.compute(i+j*out, in, out));
				tempW.setValue(i,j,0f);
			}
		}
		this.func=func;

	}

	public Blob getWeights() {
		return weights;
	}

	public void setWeights(Blob weights) {
		this.weights=weights;
	}

	public void setBias(Blob bias) {
		this.bias=bias;

	}


}