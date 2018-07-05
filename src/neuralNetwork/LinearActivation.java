package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements the linear activation per neuron
/// ----------------------------------------------------------------------------------------

public class LinearActivation implements ActivationFunction 
{
	float factor;
	float add;
	boolean use = false;
	public LinearActivation(float factor, float add) {
		this.factor = factor;
		this.add = add;
		this.use = true;
	}
	//no changes need if you wont use factor
	public LinearActivation() {
		//this.factor = factor;
	}
	// This function computes linear(x) and returns it
	public float compute(float x)
	{
		if (use) return factor*x+add;
		return x;
	}

	// This function computes linear'(x) and returns it
	public float derivative(float x) 
	{
		return 1;
	}

}
