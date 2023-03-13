#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#define e 2.71828

void calcNodeBPV(Node * node, BackPropValues bpv)
{
	node->BPV.emplace_back(bpv);
	
	for (int i = 0; i < node->parents; i++)
	{
		calcNodeBPV(node->parent[i], bpv.next(node->weight[i]));
	}

	return;
}

void calcNodeOrderAndBPV(Node *node, bool *visitedNode, Node **nodeCalculationOrder, int *connectedNodes,
						 BackPropValues bpv)
{
	node->BPV.emplace_back(bpv);

	if (!node->parents)
	{
		visitedNode[node->id] = true;
		return;
	}

	for (int i = 0; i < node->parents; i++)
	{
		if (!visitedNode[node->parent[i]->id])
		{
			calcNodeOrderAndBPV(node->parent[i], visitedNode, nodeCalculationOrder, connectedNodes,
								bpv.next(node->weight[i]));
		} else
		{
			calcNodeBPV(node->parent[i], bpv.next(node->weight[i]));
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

	outputError = new float[networkStructure.totalOutputNodes];

	for (int i = (this->networkStructure.totalNodes - this->networkStructure.totalOutputNodes);
		 i < this->networkStructure.totalNodes; i++)
	{
		BackPropValues bpv(
			&outputError[i - (this->networkStructure.totalNodes - this->networkStructure.totalOutputNodes)],
			&node[i].value);

		calcNodeOrderAndBPV(&node[i], visitedNode, nodeCalculationOrder, &connectedNodes, bpv);
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

float lazyNewWeight(float weight, float learningRate, float error)
{
	return weight - (learningRate * error * weight);
}

float NeuralNetwork::backpropagation(std::vector<float> targetValues) // FIXME slow and does redundant calculations
{
	float totalError = 0;

	for (int i = (networkStructure.totalNodes - networkStructure.totalOutputNodes); i < networkStructure.totalNodes;
		 i++)
	{
		float diff =
			node[i].value - targetValues[i - (networkStructure.totalNodes - networkStructure.totalOutputNodes)];

		outputError[i - (networkStructure.totalNodes - networkStructure.totalOutputNodes)] = diff;

		totalError += .5 * std::pow(outputError[i - (networkStructure.totalNodes - networkStructure.totalOutputNodes)], 2);
	}

	bool *visitedNode = new bool[networkStructure.totalNodes]();

	for (int i = 0; i < connectedNodes; i++)
	{
		// std::cout << *nodeCalculationOrder[i] << '\n';
		nodeCalculationOrder[i]->calcNewWeight(learningRate);
	}

	delete[] visitedNode;

	return totalError;
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
	delete[] outputError;

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
