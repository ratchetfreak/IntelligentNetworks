#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <iostream>
#include <math.h>
#include <pthread.h>

namespace in
{
	inline float dsig(float sigx)
	{
		return sigx * (1. - sigx);
	}

	class Node
	{
		public:
			int							id		= -1;
			float						value	= 0;
			int							parents = 0;
			Node					  **parent	= nullptr;
			float					  **weight	= nullptr;

			void calcNewWeight(float learningRate)
			{
				float multiplier = 0;

				for (int i = 0; i < parents; i++)
				{
					*weight[i] -= multiplier * parent[i]->value * dsig(value);
				}
			}

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

				output << "id      - " << node.id << '\n';
				output << "value   - " << node.value << '\n';
				output << "parents - " << node.parents << '\n';
				for (int i = 0; i < node.parents; i++)
				{
					output << "\tfrom    - " << node.parent[i]->id << '\n';
					output << "\tweight  - " << *node.weight[i] << '\n';
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
			const float *const		 &nodeError		   = _nodeError;

			NeuralNetwork(unsigned char* netdata, unsigned char* strudata);
			NeuralNetwork(NetworkStructure &networkStructure);
			// ~NeuralNetwork();

			void setInputNode(int nodeNumber, float value);

			void update();

			std::string serialize();

			float backpropagation(std::vector<float> targetValues);

			void destroy();
	};
} // namespace in
