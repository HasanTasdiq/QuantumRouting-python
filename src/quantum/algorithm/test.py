import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import gurobipy as gp
from gurobipy import quicksum
import networkx as nx

G = nx.cycle_graph(7)
paths = list(nx.shortest_simple_paths(G, 0, 1))
print(paths)
