package neuralNetwork;
/// ----------------------------------------------------------------------------------------
/// This class is a general container for holding multidimensional data of all kind.
/// ----------------------------------------------------------------------------------------

public class Blob 
{
	public int width;
	public int height;
	public int channels;
	public int numSize;
	public float[] values;
	
	// Create Blob with one dimension
	public Blob(int outSize) 
	{
		init( outSize,  1, 1,1);
	}
	
	// Create Blob with two dimensions
	public Blob(int width, int height) 
	{
		init( width,  height, 1,1);
	}
	
	// Create Blob with three dimensions
	public Blob(int width, int height,int channels) 
	{
		init( width,  height, channels,1);
	}
	
	// Create Blob with four dimensions
	public Blob(int width, int height,int channels, int numSize) 
	{
		init( width,  height, channels, numSize);
	}

	void init(int width, int height,int channels,int numSize)
	{
		values=new float[width*channels*height*numSize];
		this.width=width;
		this.channels=channels;
		this.height=height;
		this.numSize=numSize;
	}

	// gets value of four dimension blobs
	public float getValue(int x, int y, int channel, int num)
	{
		return values[x+y*width+channel*width*height+num*width*height*numSize];
	}
	
	// sets value of four dimension blobs
	public void setValue(int x, int y, int channel, int num, float val)
	{
		values[x+y*width+channel*width*height+num*width*height*numSize]=val;
	}
	
	// adds value of four dimension blobs
	public void addValue(int x, int y, int channel, int num, float val)
	{
		values[x+y*width+channel*width*height+num*width*height*numSize]+=val;
	}
	
	// gets value of three dimension blobs
	public float getValue(int x, int y, int channel)
	{
		return values[x+y*width+channel*width*height];
	}
	
	// ...
	public void setValue(int x, int y, int channel, float val)
	{
		values[x+y*width+channel*width*height]=val;
	}
	
	public void addValue(int x, int y, int channel, float val)
	{
		values[x+y*width+channel*width*height]+=val;
	}
	
	public float getValue(int x, int y)
	{
		return values[x+y*width];
	}
	
	public void setValue(int x, int y, float val)
	{
		values[x+y*width]=val;
	}
	
	public void addValue(int x, int y, float val)
	{
		values[x+y*width]+=val;
	}
	
	public float getValue(int idx)
	{
		return values[idx];
	}
	
	public void setValue(int idx, float val)
	{
		values[idx]=val;
	}
	
	public void addValue(int idx, float val)
	{
		values[idx]+=val;
	}
	
	public void convolve(Blob d, int channelsOut ,int windowSize, Blob weights)
	{
		int halfWindowSize=windowSize/2;
        for (int currChannel=0;currChannel<channelsOut;currChannel++)
      	{
			for(int x=halfWindowSize;x<width-halfWindowSize;x++)
			{
				for(int y=halfWindowSize;y<height-halfWindowSize;y++)
				{
					float sum=0;
					for(int c=0;c<channels;c++)
					{
						for(int i=-halfWindowSize;i<=halfWindowSize;i++)
						{
							for(int j=-halfWindowSize;j<=halfWindowSize;j++)
							{
								sum+=getValue(x+i,y+j,c)*weights.getValue(halfWindowSize+i, halfWindowSize+j, c, currChannel);
							}
						}
					}
					d.setValue(x,y,currChannel,sum);
				}
			}
      	}
	}
	
	public void convolveFullyRotatedAndAdd(Blob d, int windowSize, Blob weights)
	{
		int halfWindowSize=windowSize/2;
		for(int x=-halfWindowSize;x<width+halfWindowSize;x++)
		{
			for(int y=-halfWindowSize;y<height+halfWindowSize;y++)
			{
				
				for(int c=0;c<channels;c++)
				{
					for(int c2=0;c2<d.channels;c2++)
					{
						float sum=0;
						for(int i=-halfWindowSize;i<=halfWindowSize;i++)
						{
							for(int j=-halfWindowSize;j<=halfWindowSize;j++)
							{
								if(x+i<width && y+j<height && x+i>=0 && y+j>=0)
								{
									sum+=getValue(x+i,y+j,c)*weights.getValue(windowSize-(halfWindowSize+i)-1,windowSize-(halfWindowSize+j)-1,c2,c);
								}
							}
						}
						d.addValue(x,y,c2,sum);
					}
					
				}
				
			}
		}
	}

	// returns overall length 
	public int getLength() 
	{
		return values.length;
	}
}
