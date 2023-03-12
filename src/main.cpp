#include "../lib/ml/ml.hpp"

#include <iostream>

int main()
{
	NetworkStructure netStruct(12, 4, 2, 2,
							   {
								   {0, 4, 0.25}, //
								   {0, 5, 0.25}, //
								   {1, 6, 0.35}, //
								   {1, 7, 0.35}, //

								   {2, 4, 0.1}, //
								   {2, 5, 0.2}, //
								   {3, 4, 0.3}, //
								   {3, 5, 0.4}, //
								   {4, 6, 0.5}, //
								   {5, 6, 0.6}, //
								   {4, 7, 0.7}, //
								   {5, 7, 0.8}, //
							   });

	std::cout << netStruct << '\n';

	NeuralNetwork network(netStruct);

	network.setInputNode(0, 1);
	network.setInputNode(1, 1);
	network.setInputNode(2, 0.1);
	network.setInputNode(3, 0.5);

	network.update();

	std::cout << network.getNode(6).value << '\n';
	std::cout << network.getNode(7).value << '\n';
	std::cout << '\n';

	for (int i = 0; i < 1; i++)
	{
		network.backpropagation({0.05, 0.95});
		network.update();
		std::cout << network.getNode(6).value << '\n';
		std::cout << network.getNode(7).value << '\n';
		std::cout << '\n';
	}

	network.destroy();
}
