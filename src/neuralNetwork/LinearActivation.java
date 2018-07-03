package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements the linear activation per neuron
/// ----------------------------------------------------------------------------------------

public class LinearActivation implements ActivationFunction 
{
	// This function computes linear(x) and returns it
	public float compute(float x)
	{
		return x;
	}

	// This function computes linear'(x) and returns it
	public float derivative(float x) 
	{
		return 1;
	}

}
