import psutil
import os
from collections import OrderedDict
import sys
import networkx as nx

G=nx.MultiGraph()
G.add_node(1)
G.add_node(2)
G.add_edge(1,2)
G.add_edge(1,2)
print(G.edges)

G.remove_edge(2,1)

# dict = OrderedDict()

print(G.edges)