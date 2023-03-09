#pragma once

#include <cstdio>
#include <math.h>
#include <ostream>
#include <sstream>

class Node
{
	public:
		int	   id	   = -1;
		float  value   = 0;
		int	   parents = 0;
		Node **parent  = nullptr;
		float *weight  = 0;
};

class Connection
{
	public:
		int	  startNode;
		int	  endNode;
		float weight;
		bool  valid	 = true;
		int	  id	 = -1;
		bool  exists = true;
};

class NetworkStructure
{
	public:
		Connection *connection;
		int			totalConnections;
		int			totalInputNodes;
		int			totalHiddenNodes;
		int			totalOutputNodes;
		int			totalNodes;

		NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes, int totalOutputNodes);

		void mutate();

		friend std::ostream &operator<<(std::ostream &os, const NetworkStructure &networkStructure)
		{
			std::stringstream output;

			output << "totalConnections - " << networkStructure.totalConnections << "\n";
			output << "totalInputNodes  - " << networkStructure.totalInputNodes << "\n";
			output << "totalHiddenNodes - " << networkStructure.totalHiddenNodes << "\n";
			output << "totalOutputNodes - " << networkStructure.totalOutputNodes << "\n";
			output << "totalNodes       - " << networkStructure.totalNodes << "\n";

			output << "\nConnections\n";

			for (int i = 0; i < networkStructure.totalConnections; i++)
			{
				output << "1: " << networkStructure.connection[i].startNode << " to "
					   << networkStructure.connection[i].endNode
					   << ", weight = " << networkStructure.connection[i].weight << "\n";
			}

			return os << output.str();
		}

		~NetworkStructure();
};

class NeuralNetwork
{
	private:
		int totalNodes		 = 0;
		int totalInputNodes	 = 0;
		int totalConnections = 0;
		int connectedNodes	 = 0;

		Node	   *node;
		Node	  **inputNode;
		Connection *connection;
		Node	  **nodeCalculationOrder;

	public:
		NeuralNetwork(int totalNodes, int totalInputNodes, Connection connection[], int totalConnections);
		// ~NeuralNetwork();

		void setConnection(int connectionNumber, Connection connection);
		void setInputNode(int nodeNumber, float value);

		void update();
		void destroy();

		Connection getConnection(int connectionNumber);
		Node	   getNode(int nodeNumber);
		int		   getTotalNodes();
		int		   getTotalInputNodes();
		int		   getTotalConnections();
};
