package neuralNetwork;
import java.util.*;

public class Node
{
	private ArrayList<Double> probabilityList = new ArrayList<Double>();
	private ArrayList<Node> parents = new ArrayList<Node>();
	// Variable which determines current value of node: 0=false, 1=true
	public int currValue=1;

	public Node()
	{
		probabilityList.add(0.5);
		probabilityList.add(0.5);
	}

	public ArrayList<Node> getParents()
	{
		return parents;
	}

	public void addParent(Node n)
	{
		parents.add(n);
		int listSizeBefore=probabilityList.size();

		for(int i=0;i<listSizeBefore;i++)
		{
			probabilityList.add(0.5);
		}
	}

	public double getConditionalProbability()
	{
		int probIdx=currValue;
		int currMultiplicator=2;
		for (Node parent : parents)
		{
			probIdx+=currMultiplicator*parent.currValue;
			currMultiplicator*=2;
		}

		return probabilityList.get(probIdx);
	}

	public double computeConjunction()
	{
		double mults=getConditionalProbability();
		for (Node parent : parents)
		{
			mults*=parent.computeConjunction();
		}
		return mults;
	}
}
