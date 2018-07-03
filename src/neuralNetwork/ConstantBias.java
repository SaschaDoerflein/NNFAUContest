package neuralNetwork;

public class ConstantBias implements BiasFiller 
{
	// Computes a initial constant bias 
	public float compute(int currWeight, int in, int out) 
	{
		return 0.005f; //*currWeight*in*out;
	}

}
