
# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg

# Create a portal context.
pc = portal.Context()
pc.defineParameter("node_hw", "GPU node type", portal.ParameterType.NODETYPE, "c240g5")
pc.defineParameter("data_size", "GPU node local storage size", portal.ParameterType.STRING, "1024GB")

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()
 
# Add a raw PC to the request.
node = request.RawPC("node")
node.cores = 100
node.ram   = 500*2048

# Install and execute a script that is contained in the repository.
node.addService(pg.Execute(shell="sh", command="/local/repository/silly.sh"))

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
