import itertools
import sys
sys.path.append("../")
import networkx as nx
from topo.Topo import Topo

nodeNo = 30
r = 10
degree = 6
topo = Topo.generate(nodeNo, 0.9, 5,0.002, degree)
G = topo.updatedG()
print(G.nodes())