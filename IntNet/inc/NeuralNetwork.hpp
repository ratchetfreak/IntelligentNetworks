#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <iostream>
#include <math.h>
#include <pthread.h>

#define e 2.71828

namespace in
{
	enum ActivationFunction
	{
		sigmoid,
		tanh,
		modsig
	};

	class Node
	{
		public:
			int		id		= -1;
			float	value	= 0;
			int		parents = 0;
			Node  **parent	= nullptr;
			float **weight	= nullptr;

			std::string serialize()
			{
				unsigned char buf[4 * 3];

				intToBytes(&id, buf + (4 * 0));
				intToBytes((int *)&value, buf + (4 * 0));
				intToBytes(&parents, buf + (4 * 0));

				std::string buffer((char *)buf, 4 * 3);

				for (int i = 0; i < parents; i++)
				{
					unsigned char index[4];
					intToBytes(&parent[i]->id, index);

					buffer.append((char *)index, 4);
				}

				return buffer;
			}

			friend std::ostream &operator<<(std::ostream &os, const Node &node)
			{
				std::stringstream output;

				std::cout << "id      - " << node.id << '\n';
				std::cout << "value   - " << node.value << '\n';
				std::cout << "parents - " << node.parents << '\n';
				for (int i = 0; i < node.parents; i++)
				{
					std::cout << "\tfrom    - " << node.parent[i]->id << '\n';
					std::cout << "\tweight  - " << *node.weight[i] << '\n';
				}

				return os << output.str();
			}
	};

	class NeuralNetwork
	{
		private:
			NetworkStructure _networkStructure;

			int _connectedNodes = 0;

			Node  *_node; // INPUT HIDDEN OUTPUT
			Node **_nodeCalculationOrder;

			float *_nodeError;

			float (*activation)(float) = sig;
			float (*derivative)(float) = dsig;

			void layeredCons();
			void dynamicCons();

		public:
			Node **inputNode;
			Node  *outputNode;
			float  learningRate = 0.6;

			const NetworkStructure	 &structure			   = _networkStructure;
			const int				 &connectedNodes	   = _connectedNodes;
			const Node *const		 &node				   = _node;
			const Node *const *const &nodeCalculationOrder = _nodeCalculationOrder;
			const float *const		 &nodeError			   = _nodeError;

			NeuralNetwork(unsigned char *netdata, unsigned char *strudata);
			NeuralNetwork(NetworkStructure &networkStructure);
			// ~NeuralNetwork();

			void setActivation(ActivationFunction actfun);

			void setInputNode(int nodeNumber, float value);

			void update();

			std::string serialize();

			float backpropagation(std::vector<float> targetValues);

			void destroy();

			Node getNode(int i)
			{
				return node[i];
			}

			int getTotalNodes()
			{
				return structure.totalNodes;
			}

			Connection getConnection(int i)
			{
				return structure.connection[i];
			}
			
			static float sig(float x)
			{
				return 1. / (1. + std::pow(e, -x));
			}

			static float dsig(float sigx)
			{
				return sigx * (1. - sigx);
			}

			static float dtanh(float tanh)
			{
				return 1 - (tanh * tanh);
			}

			static float modsig(float x)
			{
				return 2 * ((1. / (1. + std::pow(e, -x))) - .5);
			}

			static float dmodsig(float modsigx)
			{
				return 2 * modsigx * (1. - modsigx);
			}
	};
} // namespace in
