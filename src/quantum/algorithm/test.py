import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import gurobipy as gp
from gurobipy import quicksum

m = gp.Model("test")
x = m.addVar(vtype=gp.GRB.BINARY, name="x")
y = m.addVar(vtype=gp.GRB.BINARY, name="y")
z = m.addVar(vtype=gp.GRB.BINARY, name="z")

m.update()

m.setObjective(x+y+2*z , gp.GRB.MAXIMIZE)

m.addConstr(quicksum([x+2,y,3*z]) <=8 , "c0")
m.addConstr(quicksum([x,y]) >=1 , "c1")


m.optimize()
m.printAttr("X")
