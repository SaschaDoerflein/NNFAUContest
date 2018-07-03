package neuralNetwork;

public class RandomWeight implements WeightFiller {

	// Computes random numbers for weight initialization
	public float compute(int currWeight, int inSize, int outSize)
	{
		return (float) (Math.random()-0.5)*0.002f;
		//return (float) (Math.random()*3f+17f); maybe travel
		//return (float) (Math.random()-0.5f); Titanic
	}

}
