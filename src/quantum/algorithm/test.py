import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import gurobipy as gp
from gurobipy import quicksum
import networkx as nx

G = nx.cycle_graph(7)
paths = [v for v in [u for u in range(10)]]
print(paths)
