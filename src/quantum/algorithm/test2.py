import networkx as nx

# Create a MultiGraph
G = nx.MultiGraph()
G.add_edge(1, 2)
G.add_edge(1, 2)
G.add_edge(2, 3)

# Get all simple paths from node 1 to node 3
path = nx.shortest_path(G, source=1, target=3)

# for path in paths:
print(path)