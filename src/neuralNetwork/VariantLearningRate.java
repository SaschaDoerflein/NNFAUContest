package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements a constant learning rate while training
/// ----------------------------------------------------------------------------------------

public class VariantLearningRate implements LearningRate 
{
	float rate;
	int iterations;
	int count;
	public VariantLearningRate(float startrate, int iterations)
	{
		this.rate = startrate;
		this.iterations = iterations+1;
		this.count = 1;
	}
	
	public float getLearningRate() 
	{
		count++;
		if (count%iterations==0) rate *= 0.999 ;
		//rate *= ((iterations-1)/iterations);
		return rate;
	}

}
