package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements a constant learning rate while training
/// ----------------------------------------------------------------------------------------

public class ConstantLearningRate implements LearningRate 
{
	float rate;
	public ConstantLearningRate(float rate)
	{
		this.rate=rate;
	}
	
	public float getLearningRate() 
	{
		return rate;
	}

}
