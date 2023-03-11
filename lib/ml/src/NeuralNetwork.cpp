#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

void travelNodeTree(Node *node, int totalNodes, Node **nodeCalculationOrder, int connectedNodes, int *nodeOrder, int id)
{
	for (int i = 0; i < totalNodes; i++)
	{
		if (node[i].parents)
		{
			for (int x = 0; x < node[i].parents; x++)
			{
				if (x == i)
				{
					continue;
				}

				if (node[i].parent[x]->id == id)
				{
					for (int y = 0; y < *nodeOrder; y++)
					{
						if (nodeCalculationOrder[y]->id == i)
						{
							goto exitLoop;
						}
					}

					nodeCalculationOrder[*nodeOrder] = &node[i];
					*nodeOrder += 1;

					travelNodeTree(node, totalNodes, nodeCalculationOrder, connectedNodes, nodeOrder, i);
				}

			exitLoop:;
			}
		}
	}

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

	int nodeOrder = 0;

	for (int i = 0; i < this->networkStructure.totalInputNodes; i++)
	{
		travelNodeTree(node, this->networkStructure.totalNodes, nodeCalculationOrder, connectedNodes, &nodeOrder, i);
	}

	connectedNodes = nodeOrder;

	return;
}

void NeuralNetwork::setInputNode(int nodeNumber, float value)
{
	inputNode[nodeNumber]->value = value;

	return;
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

		nodeCalculationOrder[i]->value = tanh(nodeCalculationOrder[i]->value);
	}

	return;
}

void NeuralNetwork::backpropagation(std::vector<float> targetValues)
{
	float *nodeError = new float[networkStructure.totalNodes];

	for (int i = (networkStructure.totalNodes - networkStructure.totalOutputNodes); i < networkStructure.totalNodes;
		 i++)
	{
		float diff =
			targetValues[i - (networkStructure.totalNodes - networkStructure.totalOutputNodes)] - node[i].value;

		nodeError[i] = diff;
	}

	for(int i = 0; i < connectedNodes; i++)
	{
		if(nodeCalculationOrder[i]->id >= (networkStructure.totalNodes - networkStructure.totalOutputNodes))
		{
			for(int x = 0; x < nodeCalculationOrder[i]->parents; x++)
			{
				*nodeCalculationOrder[i]->weight[x] = *nodeCalculationOrder[i]->weight[x] + (learningRate * nodeCalculationOrder[i]->parent[x]->value * nodeError[nodeCalculationOrder[i]->id]);
			}
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
