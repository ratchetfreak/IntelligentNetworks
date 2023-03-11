#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <math.h>

class Node
{
	public:
		int	   id	   = -1;
		float  value   = 0;
		int	   parents = 0;
		Node **parent  = nullptr;
		float **weight  = 0;
};

class NeuralNetwork
{
	private:
		NetworkStructure networkStructure;

		int connectedNodes = 0;

		Node		*node;
		Node	   **inputNode;
		Node	   **nodeCalculationOrder;

		float learningRate = 0.1;

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
