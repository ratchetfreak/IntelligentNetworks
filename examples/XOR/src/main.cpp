#include "../../../IntNet/intnet.hpp"

#include <fstream>
#include <iostream>

#include <bitset>

inline float ranfloat(float f1, float f2)
{
	return (((float)std::rand() / (float)RAND_MAX) * (std::abs(f1) + std::abs(f2))) - std::abs(f1);
}

void XORexample()
{
	in::NetworkStructure netStruct(9, 5, 2, 1,
								   {
									   {0, 5, ranfloat(-1, 1)}, //
									   {1, 6, ranfloat(-1, 1)}, //
									   {2, 7, ranfloat(-1, 1)}, //
									   {3, 5, ranfloat(-1, 1)}, //
									   {3, 6, ranfloat(-1, 1)}, //
									   {4, 5, ranfloat(-1, 1)}, //
									   {4, 6, ranfloat(-1, 1)}, //
									   {5, 7, ranfloat(-1, 1)}, //
									   {6, 7, ranfloat(-1, 1)}, //
								   });

	std::cout << netStruct << '\n';

	in::NeuralNetwork network(netStruct);

	network.setInputNode(0, 1);
	network.setInputNode(1, 1);
	network.setInputNode(2, 1);

	std::fstream fs("plot.txt", std::ios::out);

	for (int i = 0; i < 10000; i++)
	{
		network.setInputNode(3, 0);
		network.setInputNode(4, 0);
		network.update();
		fs << i << " " << network.backpropagation({0}) << '\n';

		network.setInputNode(3, 0);
		network.setInputNode(4, 1);
		network.update();
		fs << i << " " << network.backpropagation({1}) << '\n';

		network.setInputNode(3, 1);
		network.setInputNode(4, 0);
		network.update();
		fs << i << " " << network.backpropagation({1}) << '\n';

		network.setInputNode(3, 1);
		network.setInputNode(4, 1);
		network.update();
		fs << i << " " << network.backpropagation({0}) << '\n';
	}

	network.setInputNode(3, 0);
	network.setInputNode(4, 0);
	network.update();
	std::cout << network.node[7].value << '\n';

	network.setInputNode(3, 0);
	network.setInputNode(4, 1);
	network.update();
	std::cout << network.node[7].value << '\n';

	network.setInputNode(3, 1);
	network.setInputNode(4, 0);
	network.update();
	std::cout << network.node[7].value << '\n';

	network.setInputNode(3, 1);
	network.setInputNode(4, 1);
	network.update();
	std::cout << network.node[7].value << '\n';

	fs.close();

	network.destroy();
}

void XORexample2()
{
	in::NetworkStructure netStruct(3, {30, 30}, 1);

	in::NetworkStructure::randomWeights(netStruct);

	// std::cout << netStruct << '\n';

	std::cout << "starting" << '\n';

	in::NeuralNetwork network(netStruct);

	network.setInputNode(0, 1);

	std::fstream fs("plot.txt", std::ios::out);

	for (int i = 0; i < 10000; i++)
	{
		network.setInputNode(1, 0);
		network.setInputNode(2, 0);
		network.update();
		// fs << i << " " << network.backpropagation({0}) << '\n';
		network.backpropagation({0});

		network.setInputNode(1, 0);
		network.setInputNode(2, 1);
		network.update();
		// fs << i << " " << network.backpropagation({1}) << '\n';
		network.backpropagation({1});

		network.setInputNode(1, 1);
		network.setInputNode(2, 0);
		network.update();
		// fs << i << " " << network.backpropagation({1}) << '\n';
		network.backpropagation({1});

		network.setInputNode(1, 1);
		network.setInputNode(2, 1);
		network.update();
		// fs << i << " " << network.backpropagation({0}) << '\n';
		network.backpropagation({0});
	}

	network.setInputNode(1, 0);
	network.setInputNode(2, 0);
	network.update();
	std::cout << network.node[network.structure.totalNodes - 1].value << '\n';

	network.setInputNode(1, 0);
	network.setInputNode(2, 1);
	network.update();
	std::cout << network.node[network.structure.totalNodes - 1].value << '\n';

	network.setInputNode(1, 1);
	network.setInputNode(2, 0);
	network.update();
	std::cout << network.node[network.structure.totalNodes - 1].value << '\n';

	network.setInputNode(1, 1);
	network.setInputNode(2, 1);
	network.update();
	std::cout << network.node[network.structure.totalNodes - 1].value << '\n';

	fs.close();

	network.destroy();
}

void iintToBytes(int *num, unsigned char charBuff[4])
{
	charBuff[0] = *num >> (8 * 3);
	charBuff[1] = *num >> (8 * 2);
	charBuff[2] = *num >> (8 * 1);
	charBuff[3] = *num >> (8 * 0);
}

void ibytesToInt(int *num, unsigned char charBuff[4])
{
	(*num) |= (int)charBuff[0] << (8 * 3);
	(*num) |= (int)charBuff[1] << (8 * 2);
	(*num) |= (int)charBuff[2] << (8 * 1);
	(*num) |= (int)charBuff[3] << (8 * 0);
}
int main()
{
	// std::srand(time(NULL));

	// in::NetworkStructure ns(1, {2, 3}, 4);
	// in::NetworkStructure::randomWeights(ns);
	//
	// std::cout << ns << '\n';
	//
	// in::NetworkStructure ns2((unsigned char *)ns.serialize().c_str());
	//
	// std::cout << ns2 << '\n';

	// XORexample();
	// XORexample2();
	
	in::NetworkStructure ns(1, {2}, 3);
	in::NeuralNetwork nn(ns);

	std::cout << nn.structure << '\n';

	std::string nsb = ns.serialize();
	std::string nnb = nn.serialize();

	in::NeuralNetwork nn1((unsigned char*)nnb.c_str(), (unsigned char*)nsb.c_str());

	std::cout << nn1.structure << '\n';

	std::cout << '\n';

	std::cout << nn1.learningRate << '\n';
	std::cout << nn1.connectedNodes<< '\n';

	// for(int  i = 0; i < nn1.structure.totalNodes; i++)
	// {
	// 	std::cout << nn1.node[i] << '\n';
	// }

	nn.destroy();
	nn1.destroy();
}
