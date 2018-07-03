package neuralNetwork;

public class SigmoidActivation implements ActivationFunction 
	{
		// This function computes linear(x) and returns it
		public float compute(float x)
		{
			return 5 * (float) ((float)1 / ((float)1 + Math.exp(-x)));
		}

		// see https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
		public float derivative(float x) 
		{
			float sigm = compute(x);
			return 5 * sigm/((float)1-sigm);
		}
}
