#include "../inc/NetworkStructure.hpp"

#include <algorithm>
#include <math.h>
#include <vector>

void travelBranchForLoop(Connection *connection, int id, bool *isNodeVisited, int totalConnections,
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

Connection::Connection()
{
}

Connection::Connection(int start, int end, float weight)
{
	startNode	 = start;
	endNode		 = end;
	this->weight = weight;
}

NetworkStructure::NetworkStructure(const NetworkStructure &networkStructure)
{
	_totalConnections = networkStructure._totalConnections;
	_totalInputNodes  = networkStructure._totalInputNodes;
	_totalHiddenNodes = networkStructure._totalHiddenNodes;
	_totalOutputNodes = networkStructure._totalOutputNodes;
	_totalNodes		  = networkStructure._totalNodes;

	_connection = new Connection[totalConnections];

	for (int i = 0; i < totalConnections; i++)
	{
		_connection[i] = networkStructure.connection[i];
	}
}

NetworkStructure::NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes,
								   int totalOutputNodes, std::vector<Connection> connection)
{
	this->_connection		= new Connection[totalConnections];
	this->_totalConnections = totalConnections;
	this->_totalInputNodes	= totalInputNodes;
	this->_totalHiddenNodes = totalHiddenNodes;
	this->_totalOutputNodes = totalOutputNodes;
	this->_totalNodes = totalInputNodes + totalHiddenNodes + totalOutputNodes;

	for (int i = 0; i < connection.size(); i++)
	{
		this->_connection[i] = connection[i];
	}
}

void NetworkStructure::addConnection(Connection connection)
{
	for (int i = 0; i < totalConnections; i++)
	{
		if (!_connection[i].valid)
		{
			_connection[i] = connection;
		}
	}
}

void NetworkStructure::removeConnection(int index)
{
	_connection[index].valid  = false;
	_connection[index].exists = false;
}

void NetworkStructure::mutate()
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

void NetworkStructure::validate()
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

	// invalid connections going into input

	for (int i = 0; i < this->totalConnections; i++)
	{
		if (this->connection[i].valid)
		{
			if (this->connection[i].endNode < totalInputNodes)
			{
				this->_connection[i].valid = false;
			}
		}
	}

	// invalid connections going into themself
	for (int i = 0; i < this->totalConnections; i++)
	{
		if (this->connection[i].valid)
		{
			if (this->connection[i].startNode == this->connection[i].endNode)
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

NetworkStructure::~NetworkStructure()
{
	delete[] connection;
}
