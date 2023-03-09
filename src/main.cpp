#include "../lib/ml/ml.hpp"

#include <iostream>

int main()
{
	// Connection *connection = new Connection;
	// connection->weight = .5;
	// connection->startNode = 0;
	// connection->endNode = 1;
	//
	// NeuralNetwork network(2, 1, connection, 1);
	//
	// network.setInputNode(0, 1);
	//
	// network.update();
	//
	// std::cout << network.getNode(1).value << '\n';
	//
	// network.destroy();
	// delete connection;
	
	NetworkStructure netstruc(16, 2, 3, 2);

	srand(time(NULL));

	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();
	netstruc.mutate();

	std::cout << netstruc << '\n';
}
