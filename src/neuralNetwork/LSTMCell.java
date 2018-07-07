package neuralNetwork;

/** class LSTM.
*/
public class LSTMCell implements Layer
{
	// Weights for the Input Gate;
	Blob weightsInputGate;
	// Weights for the Output Gate;
	Blob weightsOutputGate;
	// Weights for the Input
	Blob weightsInputCell;
	// Weights for the Input
	Blob weightsForget;
	// Weights from net i.e output over all blocks*cells
	Blob weights;
	// Activation function Sigmoid 0-1
	ActivationFunction f;
	// Activation function Sigmoid -2-2
	ActivationFunction g;
	// Activation function Sigmoid -1-1
	ActivationFunction h;
	// Net after Input gate
	Blob netInGate;
	// Net after Output gate
	Blob netOutGate;
	// Net after Output gate before sigmoid
	Blob netOutf;
	// Net after Input
	Blob netIn;
	// Net forget
	Blob netForget;
	// Net state
	Blob state;
	// Cell Output
	Blob cellOut;
	// loss Function;
	LossFunction loss;
	// Activation function for neurons
	ActivationFunction func;
	// The output of each neuron after applying activation functions etc.
	Blob output;
	// derivative of the output units
	Blob delta;
	// derivatives of the output gates
	Blob deltaOut;
	// cell state error
	Blob err;
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
   public Blob forward(Blob inputBlob) {
	   
      for (int j = 0; j < netInGate.width;j++) {
        float sumInGate = 0f;
        float sumOutGate = 0f;
        float sumNetForget = 0f;
        for (int m = 0; m < inputBlob.getLength(); m++) {        	
			sumInGate += inputBlob.getValue(m)*weightsInputGate.getValue(j,m);
			sumOutGate += inputBlob.getValue(m)*weightsOutputGate.getValue(j,m);
			sumNetForget += inputBlob.getValue(m)*weightsForget.getValue(j,m);
        }
		
		for (int v = 0; v < weightsInputCell.channels; v++) {
			float sumIn = 0f;
			for (int m = 0; m < inputBlob.getLength(); m++) { 
				sumIn += inputBlob.getValue(m)*weightsInputCell.getValue(j,m,v);
			}
			netIn.setValue(j,v,sumIn+bias.getValue(j));
      	}
		netInGate.setValue(j,f.compute(sumInGate+bias.getValue(j)));		
		netOutf.setValue(j,sumOutGate+bias.getValue(j));
		netOutGate.setValue(j, f.compute(netOutf.getValue(j)));
		netForget.setValue(j, f.compute(sumNetForget));
		for (int v = 0; v < weightsInputCell.channels; v++) {
			state.setValue(j, v, state.getValue(j,v)*netForget.getValue(j)+netInGate.getValue(j)*g.compute(netIn.getValue(j,v)));
			cellOut.setValue(j, v, netOutGate.getValue(j)*h.compute(state.getValue(j,v)));
		}
	}
    for (int k = 0; k < output.getLength(); k ++ ) { 
      float out = 0f;
      for (int vj = 0; vj < cellOut.getLength(); vj++) {
    	  out += weights.getValue(k,vj)*cellOut.getValue(vj);
      }
      tempOut.setValue(k, out);
      output.setValue(0, f.compute(out));
    }
	return output;
   }

   public Blob backward (Blob expectedOutput, Blob weightsBefore)
   {
	  float[] lossReturn = loss.derivative(expectedOutput, output);
      for (int k=0;k<expectedOutput.getLength();k++)
      {
     		delta.setValue(k,lossReturn[k] * func.derivative(tempOut.getValue(k)));
      }
      //return delta;
      for (int j = 0; j < netOutf.getLength(); j++) {
    	  
    	  float sum2 = 0f;
    	  for (int vj = 0; vj < cellOut.getLength(); vj++) {
    		  float sum1 = 0f;
	    	  for (int k = 0; k < delta.getLength(); k++) {
	    		  sum1 += weights.getValue(k,vj)*delta.getValue(k);
	    	  }
	    	  sum2 += h.compute(state.getValue(vj))*sum1;
    	  }
    	  deltaOut.setValue(j, f.derivative(netOutf.getValue(j))*sum2);
      }
      for (int j = 0; j < netInGate.width; j++ ) {
    	  for (int v = 0; v < weightsInputCell.channels; v++) {
	    	  float sum = 0f;
	    	  for (int k = 0; k < delta.getLength(); k++) {
	    		  sum += weights.getValue(k,j*weightsInputCell.channels+v)*delta.getValue(k);
	    	  }
	    	  err.setValue(j*weightsInputCell.channels+v, netOutGate.getValue(j)*h.derivative(state.getValue(j*weightsInputCell.channels+v)*sum));
    	  }	
      }
      return delta;
   }

	public void updateWeightsAndBias(Blob inputBlob, float learningRate)
	{
		for (int vj = 0; vj < cellOut.getLength(); vj++) {
			for (int k = 0; k < delta.getLength(); k++) {
				weights.addValue(k, vj, learningRate*delta.getValue(k)*cellOut.getValue(vj));
			}
		}
		for (int j = 0; j < netInGate.width;j++) {
			for (int m = 0; m < inputBlob.getLength(); m++) {
				weightsOutputGate.addValue(j, m, learningRate*deltaOut.getValue(j)*inputBlob.getValue(m));
			}
		}
		for (int j = 0; j < netInGate.width; j++ ) {
			for (int m = 0; m < inputBlob.getLength(); m++) {
				float sum = 0f;
				for (int vj = 0; vj < cellOut.getLength(); vj++) {
					sum += err.getValue(vj)*
				}
			}
		}
		
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

	public LSTMCell(ActivationFunction func, WeightFiller fillerWeight,BiasFiller fillerBias , int in, int out, int executionPrevent)
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