#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <bitset>
#include <iostream>
#include <vector>

// NOTE - setting as visited at start of func is a lazy fix for a bigger looping
// problem will rewrite loop detection code

void calcNodeOrder(in::Node *node, bool *visitedNode, in::Node **nodeCalculationOrder, int *connectedNodes)
{
	visitedNode[node->id] = true;

	if (!node->parents)
	{
		visitedNode[node->id] = true;
		return;
	}

	for (int i = 0; i < node->parents; i++)
	{
		if (!visitedNode[node->parent[i]->id])
		{
			calcNodeOrder(node->parent[i], visitedNode, nodeCalculationOrder, connectedNodes);
		}
	}

	nodeCalculationOrder[*connectedNodes] = node;
	(*connectedNodes)++;

	return;
}

void in::NeuralNetwork::dynamicCons()
{
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

	bool *visitedNode = new bool[_networkStructure.totalNodes]();

	_nodeError = new float[_networkStructure.totalNodes];

	for (int i = (this->_networkStructure.totalNodes - this->_networkStructure.totalOutputNodes);
		 i < this->_networkStructure.totalNodes; i++)
	{
		if (!visitedNode[_node[i].id])
		{
			calcNodeOrder(&_node[i], visitedNode, _nodeCalculationOrder, &_connectedNodes);
		}
	}

	delete[] visitedNode;

	return;
}

void in::NeuralNetwork::layeredCons()
{
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

	_connectedNodes		  = structure.totalNodes - structure.totalInputNodes;
	_nodeCalculationOrder = new Node *[_connectedNodes];

	for (int i = 0; i < _connectedNodes; i++)
	{
		_nodeCalculationOrder[i] = &_node[i + structure.totalInputNodes];
	}

	_nodeError = new float[_networkStructure.totalNodes];
}

in::NeuralNetwork::NeuralNetwork(NetworkStructure &networkStructure) : _networkStructure(networkStructure)
{
	_node	  = new Node[this->_networkStructure.totalNodes];
	inputNode = new Node *[this->_networkStructure.totalInputNodes];

	outputNode = &_node[structure.totalNodes - structure.totalOutputNodes];

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

	if (structure.type == Dynamic)
	{
		dynamicCons();
	}
	else if (structure.type == Layered)
	{
		layeredCons();
	}
}

void in::NeuralNetwork::setActivation(in::ActivationFunction actfun)
{
	switch (actfun)
	{
		case in::ActivationFunction::sigmoid:
			activation = sig;
			derivative = dsig;
			break;
		case in::ActivationFunction::tanh:
			activation = std::tanh;
			derivative = dtanh;
			break;
		case in::ActivationFunction::modsig:
			activation = modsig;
			derivative = dmodsig;
			break;
	}
}

void in::NeuralNetwork::setInputNode(int nodeNumber, float value)
{
	inputNode[nodeNumber]->value = value;

	return;
}

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
		_nodeCalculationOrder[i]->value = activation(_nodeCalculationOrder[i]->value);
	}

	return;
}

void in::NeuralNetwork::updateLinearOutput()
{
	for (int i = 0; i < _connectedNodes; i++)
	{
		_nodeCalculationOrder[i]->value = 0;

		for (int x = 0; x < _nodeCalculationOrder[i]->parents; x++)
		{
			_nodeCalculationOrder[i]->value +=
				_nodeCalculationOrder[i]->parent[x]->value * (*_nodeCalculationOrder[i]->weight[x]);
		}

		if (_nodeCalculationOrder[i] < outputNode)
		{
			_nodeCalculationOrder[i]->value = activation(_nodeCalculationOrder[i]->value);
		}
	}

	return;
}

in::NeuralNetwork::NeuralNetwork(unsigned char *netdata, unsigned char *strudata) : _networkStructure(strudata)
{
	bytesToInt(&_connectedNodes, netdata + (4 * 0));
	bytesToInt((int *)&learningRate, netdata + 4 * 1);

	int offset = 2;

	_nodeError = new float[structure.totalNodes];

	_nodeCalculationOrder = new Node *[_connectedNodes];
	_node				  = new Node[structure.totalNodes];

	for (int i = 0; i < connectedNodes; i++)
	{
		int index = 0;
		bytesToInt((int *)&index, netdata + (4 * offset));

		_nodeCalculationOrder[i] = &_node[index];

		offset++;
	}

	for (int i = 0; i < structure.totalNodes; i++)
	{
		bytesToInt(&(_node[i].id), netdata + (4 * offset));
		offset++;
		bytesToInt((int *)&(_node[i].value), netdata + (4 * offset));
		offset++;
		bytesToInt(&(_node[i].parents), netdata + (4 * offset));
		offset++;

		_node[i].parent = new Node *[_node[i].parents];
		_node[i].weight = new float *[_node[i].parents];

		for (int x = 0; x < node[i].parents; x++)
		{
			int nodeIndex = 0;
			bytesToInt((int *)&(nodeIndex), netdata + (4 * offset));
			_node[i].parent[x] = &_node[nodeIndex];

			offset++;
			int weightIndex = 0;
			bytesToInt((int *)&(weightIndex), netdata + (4 * offset));

			_node[i].weight[x] = (float *)((int64_t)structure.connection.data() + weightIndex);

			offset++;
		}
	}

	inputNode  = new Node *[structure.totalInputNodes];
	outputNode = &_node[structure.totalNodes - structure.totalOutputNodes];

	// no mem issues how?
}

std::string in::NeuralNetwork::serialize()
{
	unsigned char buf[4 * 2];

	intToBytes(&_connectedNodes, buf + (4 * 0));
	intToBytes((int *)&learningRate, buf + 4 * 1);

	std::string buffer((char *)buf, 4 * 2);

	for (int i = 0; i < connectedNodes; i++)
	{
		unsigned char cb[4];

		intToBytes((int *)&_nodeCalculationOrder[i]->id, cb);
		buffer.append((char *)cb, 4);
	}

	for (int i = 0; i < structure.totalNodes; i++)
	{
    std::vector<unsigned char> storage;
    storage.resize((4 * 3) + (4 * 2 * node[i].parents));
    
		unsigned char *cb = storage.data();

		intToBytes((int *)&(node[i].id), cb);
		intToBytes((int *)&(node[i].value), cb + (4 * 1));
		intToBytes((int *)&(node[i].parents), cb + (4 * 2));

		for (int x = 0; x < node[i].parents; x++)
		{
			intToBytes((int *)&(node[i].parent[x]->id), cb + (4 * 3) + (4 * ((x * 2) + 0)));

			int64_t weightOffset = (int64_t)node[i].weight[x] - (int64_t)structure.connection.data();

			intToBytes((int *)&(weightOffset), cb + (4 * 3) + (4 * ((x * 2) + 1)));
		}

		buffer.append((char *)cb, (4 * 3) + (4 * 2 * node[i].parents));
	}

	return buffer;
}

float lazyNewWeight(float weight, float learningRate, float error)
{
	return weight - (learningRate * error * weight);
}

void in::NeuralNetwork::setupGradients(std::vector<float> *gradients)
{
	int size = 0;
	for (int i = _connectedNodes - 1; i >= 0; i--)
	{
		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			size++;
		}
	}

	(*gradients).resize(size, 0);
}

float in::NeuralNetwork::calcGradients(std::vector<float> *gradients, std::vector<float> targetValues)
{
	int index = 0;

	float totalError = 0;

	for (int i = 0; i < _networkStructure.totalNodes - _networkStructure.totalOutputNodes; i++)
	{
		_nodeError[i] = 0;
	}

	for (int i = (_networkStructure.totalNodes - _networkStructure.totalOutputNodes); i < _networkStructure.totalNodes;
		 i++)
	{
		float diff =
			_node[i].value - targetValues[i - (_networkStructure.totalNodes - _networkStructure.totalOutputNodes)];

		_nodeError[i] = diff;

		totalError += .5 * std::pow(_nodeError[i], 2);
	}

	for (int i = _connectedNodes - 1; i >= 0; i--)
	{
		// std::cout << *nodeCalculationOrder[i] << '\n';
		// _nodeCalculationOrder[i]->calcNewWeight(learningRate);

		_nodeError[nodeCalculationOrder[i]->id] *= derivative(nodeCalculationOrder[i]->value);

		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			_nodeError[nodeCalculationOrder[i]->parent[x]->id] +=
				_nodeError[nodeCalculationOrder[i]->id] * *nodeCalculationOrder[i]->weight[x];

			// w_new = w_old - learningRate * error_total * input

			(*gradients)[index] += _nodeError[nodeCalculationOrder[i]->id] * nodeCalculationOrder[i]->parent[x]->value;

			index++;
		}
	}

	return totalError;
}

void in::NeuralNetwork::applyGradients(std::vector<float> &gradients, float loss, int episodeLength)
{
	int index = 0;

	for (int i = _connectedNodes - 1; i >= 0; i--)
	{
		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			*nodeCalculationOrder[i]->weight[x] -= learningRate * (gradients[index] / episodeLength) * loss;

			index++;
		}
	}
}

float in::NeuralNetwork::backpropagation(std::vector<float> targetValues)
{
	float totalError = 0;

	for (int i = 0; i < _networkStructure.totalNodes - _networkStructure.totalOutputNodes; i++)
	{
		_nodeError[i] = 0;
	}

	for (int i = (_networkStructure.totalNodes - _networkStructure.totalOutputNodes); i < _networkStructure.totalNodes;
		 i++)
	{
		float diff =
			_node[i].value - targetValues[i - (_networkStructure.totalNodes - _networkStructure.totalOutputNodes)];

		_nodeError[i] = diff;

		totalError += .5 * std::pow(_nodeError[i], 2);
	}

	for (int i = _connectedNodes - 1; i >= 0; i--)
	{
		// std::cout << *nodeCalculationOrder[i] << '\n';
		// _nodeCalculationOrder[i]->calcNewWeight(learningRate);

		_nodeError[nodeCalculationOrder[i]->id] *= derivative(nodeCalculationOrder[i]->value);

		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			_nodeError[nodeCalculationOrder[i]->parent[x]->id] +=
				_nodeError[nodeCalculationOrder[i]->id] * *nodeCalculationOrder[i]->weight[x];

			// w_new = w_old - learningRate * error_total * input

			*nodeCalculationOrder[i]->weight[x] -=
				learningRate * _nodeError[nodeCalculationOrder[i]->id] * nodeCalculationOrder[i]->parent[x]->value;
		}
	}

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
	delete[] _nodeError;

	return;
}
