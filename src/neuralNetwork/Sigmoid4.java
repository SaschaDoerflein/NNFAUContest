package neuralNetwork;

public class Sigmoid4 implements ActivationFunction 
	{
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

		//sigmoid range -2-2
		public float compute(float x)
		{
			return (float) ((4f / (1f + Math.exp(-x)))-2f);
		}
		public float derivative(float x) 
		{
			return derivativeOld(x)*4;
		}
}
