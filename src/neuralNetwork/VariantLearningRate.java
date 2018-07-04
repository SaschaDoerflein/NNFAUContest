package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements a constant learning rate while training
/// ----------------------------------------------------------------------------------------

public class VariantLearningRate implements LearningRate 
{
	float rate;
	int iterations;
	int count;
	ActivationFunction func;
	float factor;
	public VariantLearningRate(float startrate, int iterations, float factor, ActivationFunction func)
	{
		this.rate = startrate;
		this.iterations = iterations+1;
		this.count = 1;
		this.factor = factor;
		this.func = func;
	}
	
	public float getLearningRate() 
	{
		count++;
		if (count%iterations==0) {
			rate *= factor;
			if (func != null) rate = func.compute(rate);
		}
		 
		//rate *= ((iterations-1)/iterations);
		return rate;
	}

}
