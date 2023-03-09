#include "../inc/NeuralNetwork.hpp"

#include <algorithm>
#include <vector>

NetworkStructure::NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes,
								   int totalOutputNodes)
{
	connection			   = new Connection[totalConnections];
	this->totalConnections = totalConnections;
	this->totalInputNodes  = totalInputNodes;
	this->totalHiddenNodes = totalHiddenNodes;
	this->totalOutputNodes = totalOutputNodes;
	this->totalNodes = totalInputNodes = totalHiddenNodes + totalOutputNodes;
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

		connection[index].weight = ((rand() / (float)RAND_MAX) * 2) - 1;
	}
	else if (type == 1)
	{
		int index = round((rand() / (float)RAND_MAX) * (totalConnections - 1));

		connection[index].exists = false;
	}
	else if (type == 2)
	{
		int node = -1;

		for (int x = (totalInputNodes + totalOutputNodes); x < totalNodes; x++)
		{
			node = x;

			for (int i = 0; i < totalConnections; i++)
			{
				if (!connection[i].exists)
				{
					continue;
				}

				if (connection[i].startNode == node || connection[i].endNode == node)
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

			connection[nonExistIndex].exists	= true;
			connection[nonExistIndex].startNode = node;
			connection[nonExistIndex].endNode	= connection[index].endNode;
			connection[nonExistIndex].weight	= 1;

			connection[index].endNode = node;
		}
	}
	else if (type == 3)
	{
		std::vector<int> hiddenNodes;
		hiddenNodes.reserve(totalConnections);

		for (int i = 0; i < totalConnections; i++)
		{
			if (!connection[i].exists)
			{
				continue;
			}

			if (connection[i].startNode < (totalInputNodes + totalOutputNodes))
			{
				if (std::find(hiddenNodes.begin(), hiddenNodes.end(), connection[i].startNode) != hiddenNodes.end())
				{
					hiddenNodes.emplace_back(connection[i].startNode);
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

		connection[nonExistIndex].exists	= true;
		connection[nonExistIndex].startNode = startNode;
		connection[nonExistIndex].endNode	= endNode;
		connection[nonExistIndex].weight	= ((rand() / (float)RAND_MAX) * 2) - 1;
	}
}

NetworkStructure::~NetworkStructure()
{
	delete[] connection;
}

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

NeuralNetwork::NeuralNetwork(int totalNodes, int totalInputNodes, Connection connection[], int totalConnections)
{
	this->totalNodes	   = totalNodes;
	this->totalInputNodes  = totalInputNodes;
	this->totalConnections = totalConnections;

	node			 = new Node[totalNodes];
	inputNode		 = new Node *[totalInputNodes];
	this->connection = new Connection[totalConnections];

	// link input node pointers to actual nodes
	for (int i = 0; i < totalInputNodes; i++)
	{
		inputNode[i] = &node[i];
	}

	// give every node an ID
	for (int i = 0; i < totalNodes; i++)
	{
		node[i].id = i;
	}

	// set connections stored to be equal to inputted ones
	for (int i = 0; i < totalConnections; i++)
	{
		this->connection[i].endNode	  = connection[i].endNode % totalNodes;
		this->connection[i].startNode = connection[i].startNode % totalNodes;
		this->connection[i].weight	  = connection[i].weight;
		this->connection[i].id		  = i;
		this->connection[i].valid	  = connection[i].exists;
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
						this->connection[x].valid = false;
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
				this->connection[i].valid = false;
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
				this->connection[i].valid = false;
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

			travelBranchForLoop(this->connection, i, isNodeVisited, this->totalConnections, this->totalNodes);

			delete[] isNodeVisited;
		}
	}

	delete[] isConnectionBase;

	// set the amount of parents every node has according to connection
	for (int y = 0; y < totalConnections; y++)
	{
		if (this->connection[y].valid)
		{
			node[this->connection[y].endNode].parents++;
		}
	}

	// allocate memory for every node to store a pointer to its parents
	for (int i = 0; i < totalNodes; i++)
	{
		if (node[i].parents)
		{
			node[i].parent = new Node *[node[i].parents];
			node[i].weight = new float[node[i].parents];

			int setParents = 0;

			for (int x = 0; x < totalConnections; x++)
			{
				if (this->connection[x].valid)
				{
					if (this->connection[x].endNode == i)
					{
						node[i].parent[setParents] = &node[this->connection[x].startNode];
						node[i].weight[setParents] = this->connection[x].weight;

						setParents++;
					}
				}
			}
		}
	}

	for (int i = 0; i < totalNodes; i++)
	{
		if (node[i].parents)
		{
			connectedNodes++;
		}
	}

	nodeCalculationOrder = new Node *[connectedNodes];

	int nodeOrder = 0;

	for (int i = 0; i < totalInputNodes; i++)
	{
		travelNodeTree(node, totalNodes, nodeCalculationOrder, connectedNodes, &nodeOrder, i);
	}

	connectedNodes = nodeOrder;

	for (int i = 0; i < totalConnections; i++)
	{
		connection[i].exists = this->connection[i].valid;
	}

	return;
}

void NeuralNetwork::setConnection(int connectionNumber, Connection connection)
{
	this->connection[connectionNumber] = connection;

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
		for (int x = 0; x < nodeCalculationOrder[i]->parents; x++)
		{
			nodeCalculationOrder[i]->value +=
				nodeCalculationOrder[i]->parent[x]->value * nodeCalculationOrder[i]->weight[x];
		}

		nodeCalculationOrder[i]->value = tanh(nodeCalculationOrder[i]->value);
	}

	return;
}

void NeuralNetwork::destroy()
{
	// free memory in nodes
	for (int i = 0; i < totalNodes; i++)
	{
		delete[] node[i].weight;
		delete[] node[i].parent;
	}

	delete[] node;
	node = nullptr;
	delete[] inputNode;
	inputNode = nullptr;
	delete[] connection;
	connection = nullptr;
	delete[] nodeCalculationOrder;
	nodeCalculationOrder = nullptr;

	return;
}

Connection NeuralNetwork::getConnection(int connectionNumber)
{
	return connection[connectionNumber];
}

Node NeuralNetwork::getNode(int nodeNumber)
{
	return node[nodeNumber];
}

int NeuralNetwork::getTotalNodes()
{
	return totalNodes;
}

int NeuralNetwork::getTotalInputNodes()
{
	return totalInputNodes;
}

int NeuralNetwork::getTotalConnections()
{
	return totalConnections;
}
