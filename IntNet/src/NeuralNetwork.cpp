#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#define e 2.71828

void calcNodeBPV(in::Node *node, in::BackPropValues bpv)
{
	node->BPV.emplace_back(bpv);

	for (int i = 0; i < node->parents; i++)
	{
		calcNodeBPV(node->parent[i], bpv.next(node->weight[i], &node->value));
	}

	return;
}

void calcNodeOrderAndBPV(in::Node *node, bool *visitedNode, in::Node **nodeCalculationOrder, int *connectedNodes,
						 in::BackPropValues bpv)
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
								bpv.next(node->weight[i], &node->value));
		}
		else
		{
			calcNodeBPV(node->parent[i], bpv.next(node->weight[i], &node->value));
		}
	}

	nodeCalculationOrder[*connectedNodes] = node;
	(*connectedNodes)++;
	visitedNode[node->id] = true;

	return;
}

in::NeuralNetwork::NeuralNetwork(NetworkStructure &networkStructure) : _networkStructure(networkStructure)
{
	_node	  = new Node[this->_networkStructure.totalNodes];
	inputNode = new Node *[this->_networkStructure.totalInputNodes];

	// link input node pointers to actual nodes
	for (int i = 0; i < this->_networkStructure.totalInputNodes; i++)
	{
		inputNode[i] = &_node[i];
	}

	// give every node an ID
	for (int i = 0; i < this->_networkStructure.totalNodes; i++)
	{
		_node[i].id = i;
	}

	this->_networkStructure.validate();

	// set the amount of parents every node has according to connection
	for (int y = 0; y < this->_networkStructure.totalConnections; y++)
	{
		if (this->_networkStructure.connection[y].valid)
		{
			_node[this->_networkStructure.connection[y].endNode].parents++;
		}
	}

	// allocate memory for every node to store a pointer to its parents
	for (int i = 0; i < this->_networkStructure.totalNodes; i++)
	{
		if (_node[i].parents)
		{
			_node[i].parent = new Node *[_node[i].parents];
			_node[i].weight = new float *[_node[i].parents];

			int setParents = 0;

			for (int x = 0; x < this->_networkStructure.totalConnections; x++)
			{
				if (this->_networkStructure.connection[x].valid)
				{
					if (this->_networkStructure.connection[x].endNode == i)
					{
						_node[i].parent[setParents] = &_node[this->_networkStructure.connection[x].startNode];
						_node[i].weight[setParents] = (float *)(&this->_networkStructure.connection[x].weight);

						setParents++;
					}
				}
			}
		}
	}

	for (int i = 0; i < this->_networkStructure.totalNodes; i++)
	{
		if (_node[i].parents)
		{
			_connectedNodes++;
		}
	}

	_nodeCalculationOrder = new Node *[_connectedNodes];

	_connectedNodes = 0;

	bool *visitedNode = new bool[networkStructure.totalNodes]();

	_outputError = new float[networkStructure.totalOutputNodes];

	for (int i = (this->_networkStructure.totalNodes - this->_networkStructure.totalOutputNodes);
		 i < this->_networkStructure.totalNodes; i++)
	{
		BackPropValues bpv(
			&_outputError[i - (this->_networkStructure.totalNodes - this->_networkStructure.totalOutputNodes)]);

		calcNodeOrderAndBPV(&_node[i], visitedNode, _nodeCalculationOrder, &_connectedNodes, bpv);
	}

	delete[] visitedNode;

	return;
}

void in::NeuralNetwork::setInputNode(int nodeNumber, float value)
{
	inputNode[nodeNumber]->value = value;

	return;
}

float sig(float x)
{
	return 1. / (1. + std::pow(e, -x));
}

// this is shit and can definately be improved
void in::NeuralNetwork::update()
{
	for (int i = 0; i < _connectedNodes; i++)
	{
		_nodeCalculationOrder[i]->value = 0;

		for (int x = 0; x < _nodeCalculationOrder[i]->parents; x++)
		{
			_nodeCalculationOrder[i]->value +=
				_nodeCalculationOrder[i]->parent[x]->value * (*_nodeCalculationOrder[i]->weight[x]);
		}
		// nodeCalculationOrder[i]->value = tanh(nodeCalculationOrder[i]->value);
		_nodeCalculationOrder[i]->value = sig(_nodeCalculationOrder[i]->value);
	}

	return;
}

float lazyNewWeight(float weight, float learningRate, float error)
{
	return weight - (learningRate * error * weight);
}

float in::NeuralNetwork::backpropagation(std::vector<float> targetValues) // FIXME slow and does redundant calculations
{
	float totalError = 0;

	for (int i = (_networkStructure.totalNodes - _networkStructure.totalOutputNodes); i < _networkStructure.totalNodes;
		 i++)
	{
		float diff =
			_node[i].value - targetValues[i - (_networkStructure.totalNodes - _networkStructure.totalOutputNodes)];

		_outputError[i - (_networkStructure.totalNodes - _networkStructure.totalOutputNodes)] = diff;

		totalError +=
			.5 * std::pow(_outputError[i - (_networkStructure.totalNodes - _networkStructure.totalOutputNodes)], 2);
	}

	bool *visitedNode = new bool[_networkStructure.totalNodes]();

	for (int i = 0; i < _connectedNodes; i++)
	{
		// std::cout << *nodeCalculationOrder[i] << '\n';
		_nodeCalculationOrder[i]->calcNewWeight(_learningRate);
	}

	delete[] visitedNode;

	return totalError;
}

void in::NeuralNetwork::destroy()
{
	// free memory in nodes
	for (int i = 0; i < _networkStructure.totalNodes; i++)
	{
		delete[] _node[i].weight;
		delete[] _node[i].parent;
	}

	delete[] _node;
	_node = nullptr;
	delete[] inputNode;
	inputNode = nullptr;
	delete[] _nodeCalculationOrder;
	_nodeCalculationOrder = nullptr;
	delete[] _outputError;

	return;
}
