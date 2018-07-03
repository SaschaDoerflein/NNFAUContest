package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class implements the euclidean loss, which is only used for OutputLayer
/// ----------------------------------------------------------------------------------------

public class EuclideanLoss implements LossFunction
{
	// This function computes euclideanLoss'(expected, real) and returns it
	public float[] derivative(Blob expected, Blob real)
	{
		float[] loss=new float[expected.getLength()];
		for(int i=0;i<expected.getLength();i++)
		{
			loss[i]=(expected.getValue(i)-real.getValue(i))*2;
		}
		return loss;
	}

	// This function computes euclideanLoss(expected, real) and returns it
	public float compute(Blob expected, Blob real)
	{
		float mse=0f;
		for(int i=0;i<expected.getLength();i++)
		{
			mse+=(expected.getValue(i)-real.getValue(i))*(expected.getValue(i)-real.getValue(i));
		}
		return mse/(float)expected.getLength();
	}
}
