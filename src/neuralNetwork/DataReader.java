package neuralNetwork;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.io.*;

/// ----------------------------------------------------------------------------------------
/// This class implements a few functions to do tests with
/// ----------------------------------------------------------------------------------------


public class DataReader
{
	// A very simple XOr Dataset with 2 inputs and one output
	public static Datum[] getXORData()
	{
		Blob[] in=new Blob[4];

		for(int i=0;i<in.length;i++)
		{
			in[i]=new Blob(2);
		}

		in[0].setValue(0,1);
		in[0].setValue(1,1);
		in[1].setValue(0,0);
		in[1].setValue(1,1);
		in[2].setValue(0,1);
		in[2].setValue(1,0);
		in[3].setValue(0,0);
		in[3].setValue(1,0);

		Blob[] out=new Blob[4];

		for(int i=0;i<out.length;i++)
		{
			out[i]=new Blob(1);
		}

		out[0].setValue(0,0);
		out[1].setValue(0,1);
		out[2].setValue(0,1);
		out[3].setValue(0,0);

		Datum[] xorDataset=new Datum[4];

		for(int i=0;i<4;i++)
		{
			xorDataset[i]=new Datum();
			xorDataset[i].data=in[i];
			xorDataset[i].label=out[i];
		}

		return xorDataset;
	}

	// A very simple And Dataset with 2 inputs and one output
	public static Datum[] getAndData()
	{
		Blob[] in=new Blob[4];

		for(int i=0;i<in.length;i++)
		{
			in[i]=new Blob(2);
		}

		in[0].setValue(0,1);
		in[0].setValue(1,1);
		in[1].setValue(0,0);
		in[1].setValue(1,1);
		in[2].setValue(0,1);
		in[2].setValue(1,0);
		in[3].setValue(0,0);
		in[3].setValue(1,0);

		Blob[] out=new Blob[4];

		for(int i=0;i<out.length;i++)
		{
			out[i]=new Blob(1);
		}

		out[0].setValue(0,1);
		out[1].setValue(0,0);
		out[2].setValue(0,0);
		out[3].setValue(0,0);

		Datum[] andDataset=new Datum[4];

		for(int i=0;i<4;i++)
		{
			andDataset[i]=new Datum();
			andDataset[i].data=in[i];
			andDataset[i].label=out[i];
		}

		return andDataset;
	}

	// A more advanced Dataset with 2 inputs and 4 outputs
	public static Datum[] get2In4OutDataset()
	{
		Blob[] in=new Blob[4];
		Blob[] out=new Blob[4];

		for(int i=0;i<in.length;i++)
		{
			in[i]=new Blob(2);
			out[i]=new Blob(4);
		}

		int count=0;
		for(int i=0;i<2;i++)
		{
			for(int j=0;j<2;j++)
			{
						in[count].setValue(0,i);
						in[count].setValue(1,j);

						if(i==1 && j==1)
						{
							out[count].setValue(0,1);
							out[count].setValue(1,0);
							out[count].setValue(2,0);
							out[count].setValue(3,0);
						}
						else if(i==0 && j==1)
						{
							out[count].setValue(0,0);
							out[count].setValue(1,1);
							out[count].setValue(2,0);
							out[count].setValue(3,0);
						}
						else if(i==0 && j==0)
						{
							out[count].setValue(0,0);
							out[count].setValue(1,0);
							out[count].setValue(2,1);
							out[count].setValue(3,0);
						}
						else
						{
							out[count].setValue(0,0);
							out[count].setValue(1,0);
							out[count].setValue(2,0);
							out[count].setValue(3,1);
						}

						count++;
			}
		}

		Datum[] dataset=new Datum[4];

		for(int i=0;i<dataset.length;i++)
		{
			dataset[i]=new Datum();
			dataset[i].data=in[i];
			dataset[i].label=out[i];
		}

		return dataset;
	}

	// Titanic Dataset with 6 inputs and 1 output
	// 1: Passenger class: 1st class, 2nd class, 3rd class
	// 2: 1=female 0=male
	// 3: age
	// 4: siblings of this passenger on board (brother, sister, aunt ...)
	// 5: parents+childrens of this passenger on board
	// 6: Ticket price
	public static ArrayList<Datum> readTitanicDataset(String readFile, boolean isTrainingset) throws IOException
	{
		ArrayList<Datum> data=new ArrayList<Datum>();
		try (BufferedReader br = new BufferedReader(new FileReader(readFile)))
		{
			String line;
			while ((line = br.readLine()) != null)
			{
				int count=0;
				Datum dataTemp=new Datum(6,1);
				String[] parts = line.split("\\|");

				if (parts.length < 3)
					continue;

				if(isTrainingset==true)
				{
					dataTemp.label.setValue(0, Float.parseFloat(parts[count]));
					count++;
				}
				// Passenger class 1st class, 2nd class, 3rd class
				dataTemp.data.setValue(0,Float.parseFloat(parts[count]));
				count++;

				// male/female
				if(parts[count].equals("female"))
				{
					dataTemp.data.setValue(1,1f);
				}
				else
				{
					dataTemp.data.setValue(1,0f);
				}
				count++;

				// age
				if(parts[count].equals("")) parts[count]="27";
				dataTemp.data.setValue(2,Float.parseFloat(parts[count]));
				count++;

				// siblings of this passenger on board (brother, sister, aunt ...)
				dataTemp.data.setValue(3,Float.parseFloat(parts[count]));
				count++;

				// parents+childrens of this passenger on board
				dataTemp.data.setValue(4,Float.parseFloat(parts[count]));
				count++;

				// Ticket price
				dataTemp.data.setValue(5,Float.parseFloat(parts[count]));
				data.add(dataTemp);
			}
		}
		return data;
	}

	// Traveltime Dataset with 26 inputs and 6 outputs (Regression)
	public static ArrayList<Datum> readTraveltimeDataset(String readFile, boolean isTrainingset) throws IOException
	{
		ArrayList<Datum> data=new ArrayList<Datum>();
		try (BufferedReader br = new BufferedReader(new FileReader(readFile)))
		{
			String line;
			while ((line = br.readLine()) != null)
			{
				int count=0;
				Datum dataTemp=new Datum(26,6);
				String[] parts = line.split("\\|");

				if (parts.length < 3)
					continue;

				for(int i=0;i<26;i++)
				{
					if(isTrainingset==true)
					{
						dataTemp.data.setValue(i, Float.parseFloat(parts[i]));
					}
					else
					{
						dataTemp.data.setValue(i, Float.parseFloat(parts[i+1]));
					}
				}

				if(isTrainingset==true)
				{
					for(int i=26;i<32;i++)
					{

						dataTemp.label.setValue(i-26, Float.parseFloat(parts[i]));
					}
				}
				data.add(dataTemp);
			}
		}
		return data;
	}

	// Advanced Image-Dataset with 32x32x3 (width x height x RGB) inputs and 1 output (label 1-10)
	public static ArrayList<Datum> getImageDataset(String readFile, boolean isTrainingset) throws IOException
	{
		File file = new File(readFile);
	    byte[] fileData = new byte[(int) file.length()];
	    DataInputStream dis = new DataInputStream(new FileInputStream(file));
	    dis.readFully(fileData);
	    dis.close();

	    ArrayList<Datum> data=new ArrayList<Datum>();
	    int count=0;

	    while(count<fileData.length)
	    {
	    	Datum dataTemp; //=new Datum(32,32,3,6);
			if(isTrainingset==true)
			{
				dataTemp = new Datum(32,32,3,6);
				for (int i = 0; i < 6; i++) {
					dataTemp.label.setValue(i, 0f);
					if (fileData[count]==i)	{
						dataTemp.label.setValue(i, 1f);
						System.out.println(i);
					}
				}
				count++;
			} else {
				dataTemp = new Datum(32,32,3,1);
			}
		    for(int c=0;c<3;c++)
		    {
			    for(int y=0;y<32;y++)
			    {
			    	for(int x=0;x<32;x++)
				    {
				    	dataTemp.data.setValue(x,y,c,(float)fileData[count]/255);
				    	count++;
				    }
			    }
		    }
			data.add(dataTemp);
	    }

		return data;
	}
}