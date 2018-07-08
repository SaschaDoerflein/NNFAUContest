package neuralNetwork;

public class Sigmoid2 implements ActivationFunction 
	{
		// This function computes linear(x) and returns it
		public float computeOld(float x)
		{
			return (float) ((float)1 / ((float)1 + Math.exp(-x)));
		}
		
		public float derivativeOld(float x) 
		{
			float sigm = computeOld(x);
			float result = sigm*((float)1-sigm);
			return result;
		}
		
		//sigmoid range -1-1
		public float compute(float x)
		{
			return (float) ((2f / (1f + Math.exp(-x)))-1f);
		}
		public float derivative(float x) 
		{
			return derivativeOld(x)*2;
		}
}
