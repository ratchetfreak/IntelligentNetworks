#include "../lib/ml/ml.hpp"

#include <iostream>

int main()
{
	NetworkStructure netStruct(1, 1, 0, 1, {{0, 1, .5}});

	std::cout << netStruct << '\n';

	NeuralNetwork network(netStruct);

	network.setInputNode(0, 1);

	network.update();

	std::cout << network.getNode(1).value << '\n';

	network.destroy();
}
