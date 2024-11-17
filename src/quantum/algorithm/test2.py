import networkx as nx

# Create a MultiGraph
from networkx import grid_graph
G = nx.Graph()



G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edge(1, 2)
G.add_edge(1 ,2)

print(G.nodes())
# print(len(G))
print(G.edges())
