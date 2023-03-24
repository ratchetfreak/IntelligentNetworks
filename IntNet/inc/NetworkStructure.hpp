#pragma once

#include <ostream>
#include <sstream>
#include <vector>

inline void intToBytes(int *num, unsigned char charBuff[4])
{
	charBuff[0] = *num >> (8 * 3);
	charBuff[1] = *num >> (8 * 2);
	charBuff[2] = *num >> (8 * 1);
	charBuff[3] = *num >> (8 * 0);
}

inline void bytesToInt(int *num, unsigned char charBuff[4])
{
	*num = 0;
	(*num) |= (int)charBuff[0] << (8 * 3);
	(*num) |= (int)charBuff[1] << (8 * 2);
	(*num) |= (int)charBuff[2] << (8 * 1);
	(*num) |= (int)charBuff[3] << (8 * 0);
}

namespace in
{
	class Connection
	{
		public:
			int	  startNode = -1;
			int	  endNode	= -1;
			float weight	= 0;
			bool  valid		= true;
			int	  id		= -1;
			bool  exists	= true;

			Connection();
			Connection(int start, int end, float weight);
			Connection(unsigned char *data);

			friend std::ostream &operator<<(std::ostream &os, const Connection &connection)
			{
				std::stringstream output;

				output << connection.startNode << " to " << connection.endNode << ", weight " << connection.weight;

				return os << output.str();
			}

			std::string serialize();
	};

	enum NetworkStructureType
	{
		Layered,
		Dynamic,
	};

	class NetworkStructure
	{
		private:
			int					 _totalConnections = 0;
			int					 _totalInputNodes  = 0;
			int					 _totalHiddenNodes = 0;
			int					 _totalOutputNodes = 0;
			int					 _totalNodes	   = 0;
			NetworkStructureType _type			   = Dynamic;
			Connection			*_connection	   = nullptr;
			std::vector<int>	 _hiddenLayerNodes;

		public:
			const int				   &totalConnections = _totalConnections;
			const int				   &totalInputNodes	 = _totalInputNodes;
			const int				   &totalHiddenNodes = _totalHiddenNodes;
			const int				   &totalOutputNodes = _totalOutputNodes;
			const int				   &totalNodes		 = _totalNodes;
			const Connection *const	   &connection		 = _connection;
			const NetworkStructureType &type			 = _type;
			const std::vector<int>	   &hiddenLayerNodes = _hiddenLayerNodes;

			NetworkStructure(const NetworkStructure &networkStructure);
			NetworkStructure(unsigned char *data);

			NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes, int totalOutputNodes,
							 std::vector<Connection> connection);

			NetworkStructure(int totalInputNodes, std::vector<int> totalHiddenNodes, int totalOutputNodes);

			std::string serialize();

			void addConnection(Connection connection);
			void removeConnection(int index);

			void mutate();

			static void randomWeights(NetworkStructure &networkStructure);

			void validate();

			friend std::ostream &operator<<(std::ostream &os, const NetworkStructure &networkStructure)
			{
				std::stringstream output;

				output << "totalConnections - " << networkStructure.totalConnections << "\n";
				output << "totalInputNodes  - " << networkStructure.totalInputNodes << "\n";
				output << "totalHiddenNodes - " << networkStructure.totalHiddenNodes << "\n";
				output << "totalOutputNodes - " << networkStructure.totalOutputNodes << "\n";
				output << "totalNodes       - " << networkStructure.totalNodes << "\n";

				output << "\nConnections\n";

				for (int i = 0; i < networkStructure.totalConnections; i++)
				{
					output << "1: " << networkStructure.connection[i].startNode << " to "
						   << networkStructure.connection[i].endNode
						   << ", weight = " << networkStructure.connection[i].weight << "\n";
				}

				return os << output.str();
			}

			~NetworkStructure();
	};
} // namespace in
