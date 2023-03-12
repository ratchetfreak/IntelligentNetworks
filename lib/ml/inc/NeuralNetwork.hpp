#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <math.h>

class Node
{
	public:
		int		id		= -1;
		float	value	= 0;
		int		parents = 0;
		Node  **parent	= nullptr;
		float **weight	= 0;

		friend std::ostream &operator<<(std::ostream &os, const Node &node)
		{
			std::stringstream output;

			output << "id      - " << node.id << '\n';
			output << "value   - " << node.value << '\n';
			output << "parents - " << node.parents << '\n';
			for (int i = 0; i < node.parents; i++)
			{
				output << "\tfrom    - " << node.parent[i]->id << '\n';
				output << "\tweight  - " << *node.weight[i] << '\n';
			}

			return os << output.str();
		}
};

class NeuralNetwork
{
	private:
		NetworkStructure networkStructure;

		int connectedNodes = 0;

		Node  *node; // INPUT HIDDEN OUTPUT
		Node **inputNode;
		Node **nodeCalculationOrder;

		float learningRate = 0.6;

	public:
		NeuralNetwork(NetworkStructure &networkStructure);
		// ~NeuralNetwork();

		void setInputNode(int nodeNumber, float value);

		void update();

		void backpropagation(std::vector<float> targetValues);

		void destroy();

		Connection getConnection(int connectionNumber);
		Node	   getNode(int nodeNumber);
		int		   getTotalNodes();
		int		   getTotalInputNodes();
		int		   getTotalConnections();
};
