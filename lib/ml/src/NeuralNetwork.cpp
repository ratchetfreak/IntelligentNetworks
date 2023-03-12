#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#define e 2.71828

void travelNodeTree(Node *node, bool *visitedNode, Node **nodeCalculationOrder, int *connectedNodes)
{
	if (!node->parents)
	{
		visitedNode[node->id] = true;
		return;
	}

	for (int i = 0; i < node->parents; i++)
	{
		if (!visitedNode[node->parent[i]->id])
		{
			travelNodeTree(node->parent[i], visitedNode, nodeCalculationOrder, connectedNodes);
		}
	}

	nodeCalculationOrder[*connectedNodes] = node;
	(*connectedNodes)++;
	visitedNode[node->id] = true;

	return;
}

NeuralNetwork::NeuralNetwork(NetworkStructure &networkStructure) : networkStructure(networkStructure)
{
	node	  = new Node[this->networkStructure.totalNodes];
	inputNode = new Node *[this->networkStructure.totalInputNodes];

	// link input node pointers to actual nodes
	for (int i = 0; i < this->networkStructure.totalInputNodes; i++)
	{
		inputNode[i] = &node[i];
	}

	// give every node an ID
	for (int i = 0; i < this->networkStructure.totalNodes; i++)
	{
		node[i].id = i;
	}

	this->networkStructure.validate();

	// set the amount of parents every node has according to connection
	for (int y = 0; y < this->networkStructure.totalConnections; y++)
	{
		if (this->networkStructure.connection[y].valid)
		{
			node[this->networkStructure.connection[y].endNode].parents++;
		}
	}

	// allocate memory for every node to store a pointer to its parents
	for (int i = 0; i < this->networkStructure.totalNodes; i++)
	{
		if (node[i].parents)
		{
			node[i].parent = new Node *[node[i].parents];
			node[i].weight = new float *[node[i].parents];

			int setParents = 0;

			for (int x = 0; x < this->networkStructure.totalConnections; x++)
			{
				if (this->networkStructure.connection[x].valid)
				{
					if (this->networkStructure.connection[x].endNode == i)
					{
						node[i].parent[setParents] = &node[this->networkStructure.connection[x].startNode];
						node[i].weight[setParents] = (float *)(&this->networkStructure.connection[x].weight);

						setParents++;
					}
				}
			}
		}
	}

	for (int i = 0; i < this->networkStructure.totalNodes; i++)
	{
		if (node[i].parents)
		{
			connectedNodes++;
		}
	}

	nodeCalculationOrder = new Node *[connectedNodes];

	connectedNodes = 0;

	bool *visitedNode = new bool[networkStructure.totalNodes]();

	for (int i = (this->networkStructure.totalNodes - this->networkStructure.totalOutputNodes);
		 i < this->networkStructure.totalNodes; i++)
	{
		travelNodeTree(&node[i], visitedNode, nodeCalculationOrder, &connectedNodes);
	}

	delete[] visitedNode;

	return;
}

void NeuralNetwork::setInputNode(int nodeNumber, float value)
{
	inputNode[nodeNumber]->value = value;

	return;
}

float sig(float x)
{
	return 1. / (1. + std::pow(e, -x));
}

// this is shit and can definately be improved
void NeuralNetwork::update()
{
	for (int i = 0; i < connectedNodes; i++)
	{
		nodeCalculationOrder[i]->value = 0;

		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			nodeCalculationOrder[i]->value +=
				nodeCalculationOrder[i]->parent[x]->value * (*nodeCalculationOrder[i]->weight[x]);
		}
		// nodeCalculationOrder[i]->value = tanh(nodeCalculationOrder[i]->value);
		nodeCalculationOrder[i]->value = sig(nodeCalculationOrder[i]->value);
	}

	return;
}

float dsig(float sigx)
{
	return sigx * (1. - sigx);
}

void NeuralNetwork::backpropagation(std::vector<float> targetValues)
{
	float *nodeError = new float[networkStructure.totalNodes];

	for (int i = (networkStructure.totalNodes - networkStructure.totalOutputNodes); i < networkStructure.totalNodes;
		 i++)
	{
		float diff =
			node[i].value - targetValues[i - (networkStructure.totalNodes - networkStructure.totalOutputNodes)];

		nodeError[i] = diff;
	}

	float nudge = 0;

	for (int i = (connectedNodes - 1); i >= 0; i--)
	{
		if(nodeCalculationOrder[i]->id > (networkStructure.totalNodes - networkStructure.totalOutputNodes))
		{

		}

		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			std::cout << *nodeCalculationOrder[i]->weight[x] << '\n';


			*nodeCalculationOrder[i]->weight[x] = (*nodeCalculationOrder[i]->weight[x]) -	//
												  (learningRate *							//
												   nodeError[nodeCalculationOrder[i]->id] * //
												   dsig(nodeCalculationOrder[i]->value) *	//
												   nodeCalculationOrder[i]->parent[x]->value);

			

			std::cout << nodeError[nodeCalculationOrder[i]->id] << '\n';
			std::cout << dsig(nodeCalculationOrder[i]->value) << '\n';
			std::cout << nodeCalculationOrder[i]->parent[x]->value << '\n';
			std::cout << (learningRate * nodeError[nodeCalculationOrder[i]->id] * dsig(nodeCalculationOrder[i]->value) *
						  nodeCalculationOrder[i]->parent[x]->value)
					  << '\n';
			std::cout << *nodeCalculationOrder[i]->weight[x] << '\n';
			std::cout << '\n';
		}
	}

	delete[] nodeError;

	return;
}

void NeuralNetwork::destroy()
{
	// free memory in nodes
	for (int i = 0; i < networkStructure.totalNodes; i++)
	{
		delete[] node[i].weight;
		delete[] node[i].parent;
	}

	delete[] node;
	node = nullptr;
	delete[] inputNode;
	inputNode = nullptr;
	delete[] nodeCalculationOrder;
	nodeCalculationOrder = nullptr;

	return;
}

Node NeuralNetwork::getNode(int nodeNumber)
{
	return node[nodeNumber];
}

int NeuralNetwork::getTotalNodes()
{
	return networkStructure.totalNodes;
}

int NeuralNetwork::getTotalInputNodes()
{
	return networkStructure.totalInputNodes;
}

int NeuralNetwork::getTotalConnections()
{
	return networkStructure.totalConnections;
}
