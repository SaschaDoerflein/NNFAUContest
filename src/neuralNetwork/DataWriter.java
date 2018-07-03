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
/// This class implements a few functions to write/convert data to string/file
/// ----------------------------------------------------------------------------------------


public class DataWriter
{
	public static String blobToString(Blob value)
	{
		String tmp="";
		for(int i=0;i<value.getLength();i++)
		{
			if(i>0)
			{
				tmp+="|";
			}
			tmp+=Float.toString(value.getValue(i));
		}
		return tmp;
	}
	
	public static String dataLabelsToString(ArrayList<Datum> testDatum)
	{
		String tmp="";
		for(int i=0;i<testDatum.size();i++)
		{
			tmp+=i+"|"+blobToString(testDatum.get(i).label)+"\r\n";
		}
		return tmp;
	}
	
	public static void writeLabelsToFile(String fileName ,ArrayList<Datum> testDatum)
	{
		try
		{
			File file = new File( fileName );

			// if file doesnt exists, then create it 
			if ( ! file.exists( ) )
			{
				file.createNewFile( );
			}

			FileWriter fw = new FileWriter( file.getAbsoluteFile( ) );
			BufferedWriter bw = new BufferedWriter( fw );
			bw.write( dataLabelsToString(testDatum) );
			bw.close( );
		}
		catch( IOException e )
		{
			System.out.println("Error: " + e);
			e.printStackTrace( );
		}
	}
}