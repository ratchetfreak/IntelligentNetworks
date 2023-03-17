#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <iostream>
#include <math.h>

namespace in
{

	class BackPropValues
	{
		public:
			float				*outputError;
			std::vector<float *> value;
			std::vector<float *> weight;
			BackPropValues()
			{
			}

			BackPropValues(float *outputError)
			{
				this->outputError = outputError;
			}

			BackPropValues next(float *weight, float *value)
			{
				BackPropValues bpv = *this;

				bpv.weight.emplace_back(weight);
				bpv.value.emplace_back(value);

				return bpv;
			}

			friend std::ostream &operator<<(std::ostream &os, const BackPropValues &bpv)
			{
				std::stringstream output;

				output << "outputError - " << *bpv.outputError << '\n';
				for (float *nodeValue : bpv.value)
				{
					output << "nodeValue - " << *nodeValue << '\n';
				}
				for (float *weight : bpv.weight)
				{
					output << "weight - " << *weight << '\n';
				}

				return os << output.str();
			}
	};

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
			std::vector<BackPropValues> BPV;

			void calcNewWeight(float learningRate)
			{
				float multiplier = 0;

				for (BackPropValues &bpv : BPV)
				{
					float mult = (*bpv.outputError) * learningRate;

					for (int i = 0; i < bpv.value.size(); i++)
					{
						mult *= *bpv.weight[i] * dsig(*bpv.value[i]);
					}

					multiplier += mult;
				}

				for (int i = 0; i < parents; i++)
				{
					*weight[i] -= multiplier * parent[i]->value * dsig(value);
				}
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

			float _learningRate = 0.6;

			float *_outputError;

		public:
			Node **inputNode;
			const NetworkStructure	 &structure	   = _networkStructure;
			const int				 &connectedNodes	   = _connectedNodes;
			const Node *const		 &node				   = _node;
			const Node *const *const &nodeCalculationOrder = _nodeCalculationOrder;
			const float				 &learningRate		   = _learningRate;
			const float *const		 &outputError		   = _outputError;

			NeuralNetwork(NetworkStructure &networkStructure);
			// ~NeuralNetwork();

			void setInputNode(int nodeNumber, float value);

			void update();

			float backpropagation(std::vector<float> targetValues);

			void destroy();
	};
} // namespace in
