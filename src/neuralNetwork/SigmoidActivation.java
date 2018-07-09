package neuralNetwork;

public class SigmoidActivation implements ActivationFunction 
	{
		// This function computes linear(x) and returns it
		public float compute(float x)
		{
			return (float) ((float)1 / ((float)1 + Math.exp(-x)));
		}

		// see https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
		// Sascha 4.7: Activation Functions result should be between 0 and 1
		public float derivative(float x) 
		{
			float sigm = compute(x);
			float result = sigm*((float)1-sigm);

			/*if (result<0) {
				result = 0;
			}
			if (result > 1) {
				result = 1;
			}*/
			return result;
		}
		//sigmoid range -1-1
		public float computeh(float x)
		{
			return (float) ((2f / (1f + Math.exp(-x)))-1f);
		}
		public float derivativeh(float x) 
		{
			return derivative(x)*2;
		}
		//sigmoid range -2-2
		public float computeg(float x)
		{
			return (float) ((4f / (1f + Math.exp(-x)))-2f);
		}
		public float derivativeg(float x) 
		{
			return derivative(x)*4;
		}
}
