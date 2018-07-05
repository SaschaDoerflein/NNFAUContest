package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements the InputLayer for our NeuralNetwork
/// ----------------------------------------------------------------------------------------

public class InputLayerInverse implements Layer 
{
		int inputLength;
		Blob output;
	
		// Forwards the inputBlob without changing it
	   public Blob forward(Blob inputBlob)
	   {
		  float temp;
		  for (int i = 0; i < output.values.length; i++) {
			  output.values[i] = 0f;
			  temp = inputBlob.getValue(i);
			  if (temp > 0.0000001 || temp < 0.0000001) output.values[i] = 1/inputBlob.getValue(i);
			  else output.values[i] = 1/inputBlob.getValue(i);
		  }
	      return output;
	   }
	   
	   // Backwards the inputBlob without changing it
	   public Blob backward (Blob deltaBefore, Blob weightsBefore)
	   {
	      return deltaBefore;
	   }

		public void updateWeightsAndBias(Blob inputBlob, float learningRate)
		{
			// Nothing to do
		}

		public InputLayerInverse(int in)
		{
			inputLength=in;
			output = new Blob(in);
		}
		
		public Blob getWeights() 
		{
			return null;
		}

		@Override
		public void setWeights(Blob weights) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void setBias(Blob bias) {
			// TODO Auto-generated method stub
			
		}
}
