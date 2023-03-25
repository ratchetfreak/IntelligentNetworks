#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <bitset>
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

	_outputError = new float[_networkStructure.totalOutputNodes];

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

	_outputError = new float[_networkStructure.totalOutputNodes];

	for (int i = (this->_networkStructure.totalNodes - this->_networkStructure.totalOutputNodes);
		 i < this->_networkStructure.totalNodes; i++)
	{
		BackPropValues bpv(
			&_outputError[i - (this->_networkStructure.totalNodes - this->_networkStructure.totalOutputNodes)]);

		calcNodeBPV(&_node[i], bpv);
	}
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

in::NeuralNetwork::NeuralNetwork(unsigned char *netdata, unsigned char *strudata) : _networkStructure(strudata)
{
	bytesToInt(&_connectedNodes, netdata + (4 * 0));
	bytesToInt((int *)&learningRate, netdata + 4 * 1);

	int offset = 2;

	_outputError = new float[structure.totalOutputNodes];

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
		bytesToInt((int*)&(_node[i].value), netdata + (4 * offset));
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
	
			_node[i].weight[x] = (float *)((long)structure.connection + weightIndex);

			offset++;
		}

		int size;
		bytesToInt(&size, netdata + (4 * offset));

		_node[i].BPV.resize(size);

		offset++;

		for (int x = 0; x < size; x++)
		{
			int outputIndex = 0;
			bytesToInt(&outputIndex, netdata + (4 * offset));

			_node[i].BPV[x].outputError = (float *)(outputIndex + (long)outputError);

			offset++;

			int vSize = 0;
			bytesToInt(&vSize, netdata + (4 * offset));

			_node[i].BPV[x].value.resize(vSize);

			offset++;

			int wSize = 0;
			bytesToInt(&wSize, netdata + (4 * offset));

			_node[i].BPV[x].weight.resize(wSize);

			offset++;

			for (int y = 0; y < vSize; y++)
			{
				int valueOffset = 0;
				bytesToInt(&valueOffset, netdata + (4 * offset));

				_node[i].BPV[x].value[y] = (float *)(valueOffset + (long)node);

				offset++;
			}

			for (int y = 0; y < wSize; y++)
			{
				int weightOffset;
				bytesToInt(&weightOffset, netdata + (4 * offset));

				_node[i].BPV[x].weight[y] = (float *)(weightOffset + (long)structure.connection);

				offset++;
			}
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
		unsigned char cb[(4 * 3) + (4 * 2 * node[i].parents)];

		intToBytes((int *)&(node[i].id), cb);
		intToBytes((int *)&(node[i].value), cb + (4 * 1));
		intToBytes((int *)&(node[i].parents), cb + (4 * 2));

		for (int x = 0; x < node[i].parents; x++)
		{
			intToBytes((int *)&(node[i].parent[x]->id), cb + (4 * 3) + (4 * ((x * 2) + 0)));

			int weightOffset = (long)node[i].weight[x] - (long)structure.connection;

			intToBytes((int *)&(weightOffset), cb + (4 * 3) + (4 * ((x * 2) + 1)));
		}

		buffer.append((char *)cb, (4 * 3) + (4 * 2 * node[i].parents));

		unsigned char bpvsize[4 * 1];

		int size = node[i].BPV.size();
		intToBytes(&size, bpvsize);

		buffer.append((char *)bpvsize, 4 * 1);

		for (int x = 0; x < size; x++)
		{
			unsigned char cb1[4 * 3];
			int			  offset = 0;

			int outputIndex = (long)node[i].BPV[x].outputError - (long)outputError;
			intToBytes(&outputIndex, cb1 + (4 * offset));
			offset++;

			int vSize = node[i].BPV[x].value.size();
			intToBytes(&vSize, cb1 + (4 * offset));

			offset++;

			int wSize = node[i].BPV[x].weight.size();
			intToBytes(&vSize, cb1 + (4 * offset));

			offset++;

			offset = 0;

			unsigned char cb2[4 * vSize];

			for (int y = 0; y < vSize; y++)
			{
				int valueOffset = (long)node[i].BPV[x].value[y] - (long)node;

				intToBytes(&valueOffset, cb2 + (4 * offset));
				offset++;
			}

			offset = 0;

			unsigned char cb3[4 * wSize];

			for (int y = 0; y < wSize; y++)
			{
				int weightOffset = (long)node[i].BPV[x].weight[y] - (long)structure.connection;

				intToBytes(&weightOffset, cb3 + (4 * offset));
				offset++;
			}

			buffer.append((char *)cb1, 4 * 3);
			buffer.append((char *)cb2, 4 * vSize);
			buffer.append((char *)cb3, 4 * wSize);
		}
	}

	return buffer;
}

float lazyNewWeight(float weight, float learningRate, float error)
{
	return weight - (learningRate * error * weight);
}

float in::NeuralNetwork::backpropagation(std::vector<float> targetValues)
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
		_nodeCalculationOrder[i]->calcNewWeight(learningRate);
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
