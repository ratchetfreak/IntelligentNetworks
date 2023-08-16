#include "../inc/NetworkStructure.hpp"

#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>

void travelBranchForLoop(in::Connection *connection, int id, bool *isNodeVisited, int totalConnections,
						 int totalNodes) // recursion hell
{
	isNodeVisited[connection[id].startNode] = true;

	if (isNodeVisited[connection[id].endNode])
	{
		connection[id].valid = false;

		return;
	}
	else
	{
		for (int i = 0; i < totalConnections; i++)
		{
			if (connection[i].valid == false)
			{
				continue;
			}

			if (connection[id].endNode == connection[i].startNode)
			{
				bool *isNodeVisitedBranch = new bool[totalNodes];

				for (int x = 0; x < totalNodes; x++)
				{
					isNodeVisitedBranch[x] = isNodeVisited[x];
				}

				travelBranchForLoop(connection, i, isNodeVisitedBranch, totalConnections, totalNodes);

				delete[] isNodeVisitedBranch;
			}
		}
	}

	return;
}

in::Connection::Connection()
{
}

in::Connection::Connection(int start, int end, float weight)
{
	startNode	 = start;
	endNode		 = end;
	this->weight = weight;
}

in::NetworkStructure::NetworkStructure(const NetworkStructure &networkStructure)
{
	_totalConnections = networkStructure._totalConnections;
	_totalInputNodes  = networkStructure._totalInputNodes;
	_totalHiddenNodes = networkStructure._totalHiddenNodes;
	_totalOutputNodes = networkStructure._totalOutputNodes;
	_totalNodes		  = networkStructure._totalNodes;
	_type			  = networkStructure._type;

	_hiddenLayerNodes = networkStructure._hiddenLayerNodes;

	_connection = new Connection[totalConnections];

	for (int i = 0; i < totalConnections; i++)
	{
		_connection[i] = networkStructure.connection[i];
	}
}

in::NetworkStructure::NetworkStructure(unsigned char *data)
{
	int offset = 0;

	bytesToInt(&_totalConnections, data + (4 * offset));
	offset++;
	bytesToInt(&_totalInputNodes, data + (4 * offset));
	offset++;
	bytesToInt(&_totalHiddenNodes, data + (4 * offset));
	offset++;
	bytesToInt(&_totalOutputNodes, data + (4 * offset));
	offset++;
	bytesToInt(&_totalNodes, data + (4 * offset));
	offset++;
	bytesToInt((int *)&_type, data + (4 * offset));
	offset++;

	_connection = new Connection[_totalConnections];

	for (int i = 0; i < _totalConnections; i++)
	{
		bytesToInt(&_connection[i].startNode, data + (4 * offset));
		offset++;
		bytesToInt(&_connection[i].endNode, data + (4 * offset));
		offset++;
		bytesToInt((int *)&_connection[i].weight, data + (4 * offset));
		offset++;
		bytesToInt((int *)&_connection[i].valid, data + (4 * offset));
		offset++;
		bytesToInt(&_connection[i].id, data + (4 * offset));
		offset++;
		bytesToInt((int *)&_connection[i].exists, data + (4 * offset));
		offset++;
	}

	int size = 0;
	bytesToInt(&size, data + (4 * offset));

	_hiddenLayerNodes.resize(size);

	offset++;

	for (int i = 0; i < size; i++)
	{
		int total = 0;
		bytesToInt(&total, data + (4 * offset));

		_hiddenLayerNodes[i] = total;

		offset++;
	}
}

in::NetworkStructure::NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes,
									   int totalOutputNodes, std::vector<Connection> connection)
{
	this->_connection		= new Connection[totalConnections];
	this->_totalConnections = totalConnections;
	this->_totalInputNodes	= totalInputNodes;
	this->_totalHiddenNodes = totalHiddenNodes;
	this->_totalOutputNodes = totalOutputNodes;
	this->_totalNodes		= totalInputNodes + totalHiddenNodes + totalOutputNodes;

	for (int i = 0; i < connection.size(); i++)
	{
		this->_connection[i] = connection[i];
	}
}

in::NetworkStructure::NetworkStructure(int totalInputNodes, std::vector<int> totalHiddenNodes, int totalOutputNodes,
									   bool hasBias)
{
	_type = Layered;

	this->_totalInputNodes	= totalInputNodes;
	this->_totalOutputNodes = totalOutputNodes;

	this->_hiddenLayerNodes = totalHiddenNodes;

	for (int totalHiddenNodes : totalHiddenNodes)
	{
		this->_totalHiddenNodes += totalHiddenNodes;
	}

	if (this->_totalHiddenNodes != 0)
	{
		_totalConnections = totalInputNodes * totalHiddenNodes[0];

		for (int i = 0; i < (totalHiddenNodes.size() - 1); i++)
		{
			_totalConnections += totalHiddenNodes[i] * totalHiddenNodes[i + 1];
			if (hasBias)
			{
				_totalConnections += totalHiddenNodes[i + 1];
			}
		}

		_totalConnections += (totalHiddenNodes[totalHiddenNodes.size() - 1] * totalOutputNodes);

		if (hasBias)
		{
			_totalConnections += totalOutputNodes;
		}

		this->_connection = new Connection[totalConnections];

		int i	 = 0;
		int node = 0;

		for (int x = 0; x < totalInputNodes; x++)
		{
			for (int y = 0; y < totalHiddenNodes[0]; y++)
			{
				this->_connection[i] = {x, totalInputNodes + y, 1};
				i++;
			}
		}

		node += totalInputNodes;

		for (int z = 0; z < (totalHiddenNodes.size() - 1); z++)
		{
			for (int y = 0; y < totalHiddenNodes[z + 1]; y++)
			{
				if (hasBias)
				{
					this->_connection[i] = {0, node + totalHiddenNodes[z] + y, 1};
					i++;
				}
				for (int x = 0; x < totalHiddenNodes[z]; x++)
				{
					this->_connection[i] = {node + x, node + totalHiddenNodes[z] + y, 1};
					i++;
				}
			}
			node += totalHiddenNodes[z];
		}

		for (int y = 0; y < totalOutputNodes; y++)
		{
			if (hasBias)
			{
				this->_connection[i] = {0, node + totalHiddenNodes[totalHiddenNodes.size() - 1] + y, 1};
				i++;
			}
			for (int x = 0; x < totalHiddenNodes[totalHiddenNodes.size() - 1]; x++)
			{
				this->_connection[i] = {node + x, node + totalHiddenNodes[totalHiddenNodes.size() - 1] + y, 1};
				i++;
			}
		}
	}
	else
	{
		_totalConnections = totalInputNodes * totalOutputNodes;

		this->_connection = new Connection[totalConnections];

		for (int x = 0; x < totalInputNodes; x++)
		{
			for (int y = 0; y < totalOutputNodes; y++)
			{
				this->_connection[(x * totalOutputNodes) + y] = {x, totalInputNodes + y, 1};
			}
		}
	}

	_totalNodes = this->totalInputNodes + this->totalHiddenNodes + this->totalOutputNodes;
}

std::string in::NetworkStructure::serialize()
{
	unsigned char buf[(6 * 4)];

	intToBytes(&_totalConnections, buf + (4 * 0));
	intToBytes(&_totalInputNodes, buf + (4 * 1));
	intToBytes(&_totalHiddenNodes, buf + (4 * 2));
	intToBytes(&_totalOutputNodes, buf + (4 * 3));
	intToBytes(&_totalNodes, buf + (4 * 4));
	intToBytes((int *)&_type, buf + (4 * 5));

	std::string buffer((char *)buf, (6 * 4));

	for (int i = 0; i < _totalConnections; i++)
	{
		unsigned char cb[4 * 6];

		intToBytes(&_connection[i].startNode, cb + (4 * 0));
		intToBytes(&_connection[i].endNode, cb + (4 * 1));
		intToBytes((int *)&_connection[i].weight, cb + (4 * 2));
		intToBytes((int *)&_connection[i].valid, cb + (4 * 3));
		intToBytes(&_connection[i].id, cb + (4 * 4));
		intToBytes((int *)&_connection[i].exists, cb + (4 * 5));

		buffer.append((char *)cb, 4 * 6);
	}

	unsigned char cb1[4 * 1];

	int size = _hiddenLayerNodes.size();
	intToBytes(&size, cb1);

	buffer.append((char *)cb1, 4 * 1);

	for (int i = 0; i < size; i++)
	{
		unsigned char cb2[4 * 1];

		int total = _hiddenLayerNodes[i];
		intToBytes(&total, cb2);

		buffer.append((char *)cb2, 4 * 1);
	}

	return buffer;
}

void in::NetworkStructure::addConnection(Connection connection)
{
	for (int i = 0; i < totalConnections; i++)
	{
		if (!_connection[i].valid)
		{
			_connection[i] = connection;
		}
	}
}

void in::NetworkStructure::removeConnection(int index)
{
	_connection[index].valid  = false;
	_connection[index].exists = false;
}

void in::NetworkStructure::mutate()
{
	int nonExistIndex = -1;

	for (int i = 0; i < totalConnections; i++)
	{
		if (!connection[i].exists)
		{
			nonExistIndex = i;
			break;
		}
	}

	int max = 3;

	if (nonExistIndex == -1)
	{
		max = 1;
	}

	int type = round((rand() / (float)RAND_MAX) * max);

	// 0 - mutate weight
	// 1 - remove connection
	// 2 - Add node
	// 3 - Add connection

	if (type == 0)
	{
		int index = round((rand() / (float)RAND_MAX) * (totalConnections - 1));
		int start = connection[index].startNode;
		int end	  = connection[index].endNode;

		_connection[index].weight = ((rand() / (float)RAND_MAX) * 2) - 1;
	}
	else if (type == 1)
	{
		int index = round((rand() / (float)RAND_MAX) * (totalConnections - 1));

		_connection[index].exists = false;
	}
	else if (type == 2)
	{
		int node = -1;

		for (int x = (totalInputNodes + totalOutputNodes); x < totalNodes; x++)
		{
			node = x;

			for (int i = 0; i < totalConnections; i++)
			{
				if (!_connection[i].exists)
				{
					continue;
				}

				if (_connection[i].startNode == node || connection[i].endNode == node)
				{
					node = -1;
				}
			}

			if (node != -1)
			{
				break;
			}
		}

		if (node != -1)
		{
			int index = round((rand() / (float)RAND_MAX) * (totalConnections - 1));

			_connection[nonExistIndex].exists	 = true;
			_connection[nonExistIndex].startNode = node;
			_connection[nonExistIndex].endNode	 = connection[index].endNode;
			_connection[nonExistIndex].weight	 = 1;

			_connection[index].endNode = node;
		}
	}
	else if (type == 3)
	{
		std::vector<int> hiddenNodes;
		hiddenNodes.reserve(totalConnections);

		for (int i = 0; i < totalConnections; i++)
		{
			if (!_connection[i].exists)
			{
				continue;
			}

			if (_connection[i].startNode < (totalInputNodes + totalOutputNodes))
			{
				if (std::find(hiddenNodes.begin(), hiddenNodes.end(), _connection[i].startNode) != hiddenNodes.end())
				{
					hiddenNodes.emplace_back(_connection[i].startNode);
				}
			}
		}

		int startNode = round((rand() / (float)RAND_MAX) * (totalInputNodes + hiddenNodes.size() - 1));
		int endNode	  = round((rand() / (float)RAND_MAX) * (totalOutputNodes + hiddenNodes.size() - 1));

		if (startNode >= totalInputNodes)
		{
			startNode -= totalInputNodes;
			startNode = hiddenNodes[startNode];
		}

		if (endNode >= totalOutputNodes)
		{
			endNode -= totalOutputNodes;
			endNode = hiddenNodes[startNode];
		}
		else
		{
			endNode += totalInputNodes;
		}

		_connection[nonExistIndex].exists	 = true;
		_connection[nonExistIndex].startNode = startNode;
		_connection[nonExistIndex].endNode	 = endNode;
		_connection[nonExistIndex].weight	 = ((rand() / (float)RAND_MAX) * 2) - 1;
	}
}

void in::NetworkStructure::randomWeights(NetworkStructure &networkStructure)
{
	for (int i = 0; i < networkStructure.totalConnections; i++)
	{
		networkStructure._connection[i].weight = ((std::rand() / (float)RAND_MAX) * 2) - 1;
	}
}

void in::NetworkStructure::validate()
{
	// give every connection an id
	for (int i = 0; i < totalConnections; i++)
	{
		_connection[i].id = i;
	}

	// Invalidate duplicated connections

	for (int i = 0; i < this->totalConnections; i++)
	{
		if (this->connection[i].valid)
		{
			for (int x = 0; x < this->totalConnections; x++)
			{
				if (this->connection[x].valid && x != i)
				{
					if (this->connection[x].startNode == this->connection[i].startNode &&
						this->connection[x].endNode == this->connection[i].endNode)
					{
						this->_connection[x].valid = false;
					}
				}
			}
		}
	}

	// invalidate connections depending on input and output

	for (int i = 0; i < this->totalConnections; i++)
	{
		if (this->connection[i].valid)
		{
			if (this->connection[i].endNode < totalInputNodes) // going into input
			{
				this->_connection[i].valid = false;
			}

			if (this->connection[i].startNode == this->connection[i].endNode) // going into self
			{
				this->_connection[i].valid = false;
			}

			// going to or from negative or going to an id too high

			if (this->connection[i].startNode < 0 || this->connection[i].endNode < 0)
			{
				this->_connection[i].valid = false;
			}

			if (this->connection[i].startNode > totalNodes || this->connection[i].endNode > totalNodes)
			{
				this->_connection[i].valid = false;
			}
		}
	}

	// Invalidate looping connections

	bool *isConnectionBase = new bool[(unsigned int)(this->totalConnections)];

	for (int i = 0; i < totalConnections; i++)
	{
		if (!this->connection[i].valid)
		{
			isConnectionBase[i] = false;

			continue;
		}

		int startNode = this->connection[i].startNode;

		bool isNodeBase = true;

		for (int x = 0; x < totalConnections; x++)
		{
			if (x == i)
			{
				continue;
			}

			if (this->connection[x].endNode == startNode)
			{
				isNodeBase = false;
				break;
			}
		}

		isConnectionBase[i] = isNodeBase;
	}

	for (int i = 0; i < totalConnections; i++)
	{
		if (isConnectionBase[i])
		{
			bool *isNodeVisited = new bool[totalNodes];

			for (int i = 0; i < this->totalNodes; i++)
			{
				isNodeVisited[i] = false;
			}

			travelBranchForLoop(this->_connection, i, isNodeVisited, this->_totalConnections, this->_totalNodes);

			delete[] isNodeVisited;
		}
	}

	delete[] isConnectionBase;

	for (int i = 0; i < totalConnections; i++)
	{
		_connection[i].exists = _connection[i].valid;
	}
}

in::NetworkStructure::~NetworkStructure()
{
	delete[] connection;
}
