package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements the Output-Layer with weights and bias.
/// ----------------------------------------------------------------------------------------

public class OutputLayer implements Layer
{
	// Loss function used for comparing output with predicted values
	LossFunction loss;
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
        output.values[j]=0f;
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

   
	public void updateWeightsAndBias(Blob inputBlob, float learningRate)
	{
		for (int j=0;j<neuronDelta.getLength();j++)
	      {
		      for (int i=0;i<inputBlob.getLength();i++)
		      {
		     		tempW.addValue(i,j,neuronDelta.getValue(j)*inputBlob.getValue(i)) ;
		      }
		      tempB.addValue(j, neuronDelta.getValue(j));
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
	/*/
   /*
   public void updateWeightsAndBias(Blob inputBlob, float learningRate)
	{
		for (int j=0;j<neuronDelta.getLength();j++)
	      {
		      for (int i=0;i<inputBlob.getLength();i++)
		      {
		     		weights.addValue(i,j,neuronDelta.getValue(j)*inputBlob.getValue(i)*learningRate) ;
		      }
		      bias.addValue(j, neuronDelta.getValue(j)*learningRate);
	      }
	}
   */
   public Blob backward (Blob expectedOutput, Blob weightsBefore)
   {
	  float[] lossReturn = loss.derivative(expectedOutput, output);
      for (int i=0;i<expectedOutput.getLength();i++)
      {
     		neuronDelta.setValue(i,lossReturn[i] * func.derivative(tempOut.getValue(i)));
      }
      return neuronDelta;
   }

   /** Constructor. */
   public OutputLayer(LossFunction loss, ActivationFunction func, WeightFiller fillerWeight, BiasFiller fillerBias, int in, int out, int executionPrevent)
   {
      this.executionPrevent=executionPrevent;
      execution = 0;
     		tempW=new Blob(in,out);
      	tempB=new Blob(out);
     		tempOut=new Blob(out);
		output=new Blob(out);
		neuronDelta=new Blob(out);
		weights=new Blob(in,out);
		bias=new Blob(out);

		for(int i=0;i<bias.getLength();i++)
		{
		   float temp=fillerBias.compute(i, in, out);
			bias.setValue(i,temp);
			tempB.setValue(i, 0f);
		}

		for(int i=0;i<in;i++)
		{
			for(int j=0;j<out;j++)
			{
			   	float temp=fillerWeight.compute(i*out+j, in, out);
				weights.setValue(i, j,temp);
				tempW.setValue(i,j,0f);
			}
		}

	   this.loss=loss;
	   this.func=func;
   }

	public Blob getWeights() {
		return weights;
	}

	@Override
	public void setWeights(Blob weights) {
		this.weights=weights;
	}

	public void setBias(Blob bias) {
		this.bias=bias;

	}
}