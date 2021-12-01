import flwr as fl
from fmore.aggregator.strategy import fedAuction as fedA

num_rounds = 2
strategy = fedA.FedAuction()

# Set up a logger for saving the logs into a separate file
#fl.common.logger.configure("server", filename='Logs/logfile')

# Exposes the server by default on port 8080
fl.server.start_server("localhost:8080",
                       strategy=strategy,
                       config={"num_rounds": num_rounds})
