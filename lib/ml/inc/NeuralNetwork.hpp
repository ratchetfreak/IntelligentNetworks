#pragma once

#include "NetworkStructure.hpp"

#include <cstdio>
#include <iostream>
#include <math.h>

class BackPropValues
{
	public:
		float				*outputError;
		float				*outputValue;
		float *inputValue;
		std::vector<float *> weight;

		BackPropValues()
		{
		}

		BackPropValues(float *outputError, float *outputValue)
		{
			this->outputError = outputError;
			this->outputValue = outputValue;
		}

		BackPropValues next(float *weight)
		{
			BackPropValues bpv = *this;

			bpv.weight.emplace_back(weight);

			return bpv;
		}

		friend std::ostream &operator<<(std::ostream &os, const BackPropValues &bpv)
		{
			std::stringstream output;

			output << "outputError - " << *bpv.outputError << '\n';
			output << "outputValue - " << *bpv.outputValue << '\n';
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
				float mult = (*bpv.outputError) * dsig(*bpv.outputValue) * learningRate;

				for(float *weight : bpv.weight)
				{
					mult *= *weight;
				}

				multiplier += mult;
			}

			for (int i = 0; i < parents; i++)
			{
				// std::cout << "old " << *weight[i] << '\n';
				*weight[i] -= multiplier * parent[i]->value * dsig(value);
				// std::cout << "new " << *weight[i] << '\n';
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
		NetworkStructure networkStructure;

		int connectedNodes = 0;

		Node  *node; // INPUT HIDDEN OUTPUT
		Node **inputNode;
		Node **nodeCalculationOrder;

		float learningRate = 0.6;

		float *outputError;

	public:
		NeuralNetwork(NetworkStructure &networkStructure);
		// ~NeuralNetwork();

		void setInputNode(int nodeNumber, float value);

		void update();

		float backpropagation(std::vector<float> targetValues);

		void destroy();

		Connection getConnection(int connectionNumber);
		Node	   getNode(int nodeNumber);
		int		   getTotalNodes();
		int		   getTotalInputNodes();
		int		   getTotalConnections();
};
