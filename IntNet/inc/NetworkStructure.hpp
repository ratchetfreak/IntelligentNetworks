#pragma once

#include <ostream>
#include <sstream>
#include <vector>

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

			friend std::ostream &operator<<(std::ostream &os, const Connection &connection)
			{
				std::stringstream output;

				output << connection.startNode << " to " << connection.endNode << ", weight " << connection.weight;

				return os << output.str();
			}
	};

	enum NetworkStructureType
	{
		Layered,
		Dynamic,
	};

	class NetworkStructure
	{
		private:
			Connection *_connection		  = nullptr;
			int			_totalConnections = 0;
			int			_totalInputNodes  = 0;
			int			_totalHiddenNodes = 0;
			int			_totalOutputNodes = 0;
			int			_totalNodes		  = 0;

		public:
			const Connection *const &connection		  = _connection;
			const int				&totalConnections = _totalConnections;
			const int				&totalInputNodes  = _totalInputNodes;
			const int				&totalHiddenNodes = _totalHiddenNodes;
			const int				&totalOutputNodes = _totalOutputNodes;
			const int				&totalNodes		  = _totalNodes;

			NetworkStructure(const NetworkStructure &networkStructure);

			NetworkStructure(int totalConnections, int totalInputNodes, int totalHiddenNodes, int totalOutputNodes,
							 std::vector<Connection> connection);

			NetworkStructure(int totalInputNodes, std::vector<int> totalHiddenNodes, int totalOutputNodes);

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
