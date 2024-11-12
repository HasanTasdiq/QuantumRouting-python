import networkx as nx

# Create a MultiGraph
from networkx import grid_graph
G = nx.grid_2d_graph(5,5)



# G = nx.Graph()
# G.add_edge(1, 2)
# # G.add_edge(2, 3)
mapping = {}
k = 0
for i in range(5):
    for j in range(5):
        mapping[(i , j)] = k
        k+=1
        G.nodes[(i , j)]["pos"] = (i,j)
# print(mapping)
pos = {(x, y): (x, y) for x, y in G.nodes()}
# G.pos = pos
G = nx.relabel_nodes(G, mapping)


# # Create a mapping for the new node names
# mapping = {1: 'A', 2: 'B', 3: 'C'}

# # Relabel the nodes
# G = nx.relabel_nodes(G, mapping)

print(G.nodes())
# print(len(G))
print(G.edges())
print(G.get_node_attributes(pos))