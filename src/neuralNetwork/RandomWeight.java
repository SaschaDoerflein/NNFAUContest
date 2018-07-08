package neuralNetwork;

public class RandomWeight implements WeightFiller {

	// Computes random numbers for weight initialization
	public float compute(int currWeight, int inSize, int outSize)
	{
		return (float) ((Math.random()-0.5));
	}

}
