package neuralNetwork;

public class TanhActivation implements ActivationFunction {

	// This method implements tanh(x)
	public float compute(float x) 
	{
		return (float) Math.tanh(x);
	}

	// This method implements tanh'(x)
	public float derivative(float x) 
	{
		final float temp=(float) Math.tanh(x);
		return 1-temp*temp;
	}

}
