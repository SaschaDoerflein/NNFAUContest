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
	ActivationFunction linear;
	// Net after Input gate
	Blob netInGate;
	// Net after Input gate befor sigmoid
	Blob netInGatef;
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
	//ActivationFunction func;
	// The output of each neuron after applying activation functions etc.
	Blob output;
	// derivative of the output units
	Blob delta;
	// derivatives of the output gates
	Blob deltaOut;
	// cell state error
	Blob err;
	//derivative s in direction w_in_m
	Blob deltaIn;
	//derivative s in direction w_c_m
	Blob deltaCell;
	// Bias for each neuron
	Blob bias;
	// You may need a temporary variable for partially processed output (if you do not need this variable, just ignore it)
	Blob tempOut;
	//reset state, deltaCell, deltaIn
	int reset;
	//reset counter
	int counter;
	
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
		netInGatef.setValue(j,sumInGate+bias.getValue(j));
		netInGate.setValue(j,f.compute(netInGatef.getValue(j)));		
		netOutf.setValue(j,sumOutGate+bias.getValue(j));
		netOutGate.setValue(j, f.compute(netOutf.getValue(j)));
		netForget.setValue(j, f.compute(sumNetForget));
		for (int v = 0; v < weightsInputCell.channels; v++) {
			//state.setValue(j, v, state.getValue(j,v)*netForget.getValue(j)+netInGate.getValue(j)*g.compute(netIn.getValue(j,v)));
			state.addValue(j, v, netInGate.getValue(j)*g.compute(netIn.getValue(j,v)));
			cellOut.setValue(j, v, netOutGate.getValue(j)*h.compute(state.getValue(j,v)));
		}
	}
    for (int k = 0; k < output.getLength(); k ++ ) { 
      float out = 0f;      
      for (int m = 0; m < inputBlob.getLength(); m++) {
    	  out += weights.getValue(k,m)*inputBlob.getValue(m);
      }
      for (int j = 0; j < netInGate.width;j++) {    	  
	      for (int v = 0; v < weightsInputCell.channels; v++) {
	    	  out += weights.getValue(k,inputBlob.getLength()+j+v*netInGate.width)*cellOut.getValue(j,v);		      
    	  }
      }
      tempOut.setValue(k, out);
      output.setValue(k, f.compute(out));
      output.setValue(k, linear.compute(output.getValue(k)));
    }
	return output;
   }

   public Blob backward (Blob expectedOutput, Blob weightsBefore)
   {
	  if (weightsBefore != null) {
		  if (weightsBefore.getValue(0) == 1) {
			  reset();
			  return null;
		  }
	  }
	  float[] lossReturn = loss.derivative(expectedOutput, output);
      for (int k=0;k<expectedOutput.getLength();k++)
      {
     		delta.setValue(k,lossReturn[k] * f.derivative(tempOut.getValue(k)));
      }
      for (int j = 0; j < netOutf.getLength(); j++) {
    	  
    	  float sum2 = 0f;
		  for (int v = 0; v < weightsInputCell.channels; v++) {
    		  float sum1 = 0f;
	    	  for (int k = 0; k < delta.getLength(); k++) {
	    		  sum1 += weights.getValue(k,weightsInputGate.height+j+v*netInGate.width)*delta.getValue(k);
	    	  }
	    	  sum2 += h.compute(state.getValue(j,v))*sum1;
		  }
    	  deltaOut.setValue(j, f.derivative(netOutf.getValue(j))*sum2);
      }
      for (int j = 0; j < netInGate.width; j++ ) {
    	  for (int v = 0; v < weightsInputCell.channels; v++) {
	    	  float sum = 0f;
	    	  for (int k = 0; k < delta.getLength(); k++) {
	    		  sum += weights.getValue(k,weightsInputGate.height+j+v*netInGate.width)*delta.getValue(k);
	    	  }
	    	  err.setValue(j,v, netOutGate.getValue(j)*h.derivative(state.getValue(j,v)*sum));
    	  }	
      }
      return delta;
   }

	public void updateWeightsAndBias(Blob inputBlob, float learningRate)
	{
		for (int k = 0; k < output.getLength(); k ++ ) { 
		     for (int m = 0; m < inputBlob.getLength(); m++) {
		    	 weights.addValue(k, m, learningRate*delta.getValue(k)*inputBlob.getValue(m));
		     }
		     for (int j = 0; j < netInGate.width;j++) {    	  
			     for (int v = 0; v < weightsInputCell.channels; v++) {
			    	 weights.addValue(k, inputBlob.getLength()+j+v*netInGate.width, learningRate*delta.getValue(k)*cellOut.getValue(j,v));		      
		    	 }
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
				for (int v = 0; v < weightsInputCell.channels; v++) {
					deltaIn.addValue(j,m,v,g.compute(netIn.getValue(j,v))*f.derivative(netInGatef.getValue(j))*inputBlob.getValue(m));
					sum += err.getValue(j,v)*deltaIn.getValue(j,m,v);
					deltaCell.addValue(j,m,v, g.derivative(netIn.getValue(j,v))*f.compute(netInGatef.getValue(j))*inputBlob.getValue(m));
					weightsInputCell.addValue(j, m, v, learningRate*err.getValue(j,v)*deltaCell.getValue(j,m)); 
				}
				weightsInputGate.addValue(j, m, learningRate*sum);
			}
		}
		/*counter++;
		if (counter == reset) {
			for (int i = 0; i < state.getLength(); i++) {
				state.setValue(i, 0);
			}
			for (int i = 0; i < deltaCell.getLength(); i++) {
				deltaCell.setValue(i, 0);
				deltaIn.setValue(i, 0);
			}
			counter = 0;
		}*/
	}
	
	public void reset() {
		for (int i = 0; i < state.getLength(); i++) {
			state.setValue(i, 0);
		}
		for (int i = 0; i < deltaCell.getLength(); i++) {
			deltaCell.setValue(i, 0);
			deltaIn.setValue(i, 0);
		}
	}

	public LSTMCell(WeightFiller fillerWeight, BiasFiller fillerBias , int blocks, int cells,  int in, int out)
	{
		this.reset = in+out-1;
		this.counter = 0;
		this.linear = new LinearActivation(0.3f,0f);
		this.f = new SigmoidActivation();
		this.g = new Sigmoid2();
		this.h = new Sigmoid4();
		this.bias = new Blob(blocks);
		for(int i=0;i<bias.getLength();i++) {
			bias.setValue(i,fillerBias.compute(i, in, out));
		}
		this.weightsInputGate = new Blob(blocks,in);
		this.weightsOutputGate = new Blob(blocks,in);
		this.weightsInputCell = new Blob(blocks,in,cells);
		this.weightsForget = new Blob(blocks,in);
		this.weights = new Blob(out,in+blocks*cells);
		this.netInGate = new Blob(blocks);		
		this.netInGatef = new Blob(blocks);		
		this.netOutGate = new Blob(blocks);		
		this.netOutf = new Blob(blocks);		
		this.netIn = new Blob(blocks,cells);		
		this.netForget = new Blob(blocks);
		this.state = new Blob(blocks,cells);
		this.cellOut = new Blob(blocks,cells);
		this.loss = new EuclideanLoss();
		this.delta = new Blob(out);
		this.deltaOut = new Blob(blocks);
		this.err = new Blob(blocks,cells);
		this.deltaIn = new Blob(blocks,in,cells);
		this.deltaCell = new Blob(blocks,in,cells);
		output=new Blob(out);
		this.tempOut = new Blob(out);
		bias=new Blob(blocks);
		for(int j=0;j<blocks;j++) {
			for(int m=0;m<in;m++) {
				weightsInputGate.setValue(j,m,fillerWeight.compute(j+m*blocks, j, m));
				weightsOutputGate.setValue(j,m,fillerWeight.compute(j+m*blocks, j, m));
				weightsForget.setValue(j,m,fillerWeight.compute(j+m*blocks, j, m));
				for (int v = 0; v < cells; v++) {
					weightsInputCell.setValue(j,m,v,fillerWeight.compute(j+m*blocks+v*blocks*in, j, m));
					
				}
			}					
		}
		for (int k = 0; k < out; k++) {
			for(int m=0;m<in;m++) {
			weights.setValue(k,m,fillerWeight.compute(k+m*out, k, m));
			for(int j=0;j<blocks;j++) {
				for (int v = 0; v < cells; v++) {				
					weights.setValue(k,in+j+v*blocks,fillerWeight.compute(k+(in+j+v*blocks)*out, j, m));
				}
			}
				
			}
		}
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