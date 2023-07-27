import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import gurobipy as gp
from gurobipy import quicksum
import networkx as nx

G = nx.cycle_graph(7)
paths = [9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11, 11, 52, 52, 52, 52, 52, 52, 52, 74, 74, 74, 76, 76, 76, 76, 76, 76, 86, 86, 86, 86, 86, 86, 86, 88, 88, 88, 93, 93, 93, 93]

print([p for p in set(paths)])


