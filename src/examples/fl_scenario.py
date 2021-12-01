import os, sys
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import tensorflow as tf
import numpy as np
import flwr as fl

from typing import Tuple
from multiprocessing import Process

from fmore.aggregator.strategy import fedAuction as fedA
from fmore.node import fmoreClient as fC


# HiddenPrints represses all contained print() statements.
# use like:
# with HiddenPrints():
#    Code...
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Used for type signatures
DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

from typing import List, Tuple, cast

import tensorflow as tf
import numpy as np

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split x and y into a number of partitions."""
    return list(
        zip(np.array_split(x, num_partitions),
            np.array_split(y, num_partitions)))


def create_partitions(
    source_dataset: XY,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a source dataset."""
    x, y = source_dataset
    x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions


def load(num_partitions: int, ) -> PartitionedDataset:
    """Create partitioned version of CIFAR-10."""
    xy_train, xy_test = tf.keras.datasets.cifar10.load_data()

    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)

    return list(zip(xy_train_partitions, xy_test_partitions))


def start_server(num_rounds: int, num_clients: int):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = fedA.FedAuction(min_available_clients=num_clients)

    # Set up a logger for saving the logs into a separate file
    fl.common.logger.configure("server", filename='./logfile')

    # Exposes the server by default on port 8080
    fl.server.start_server("localhost:8080",
                           strategy=strategy,
                           config={"num_rounds": num_rounds})


def start_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10

    model = tf.keras.applications.MobileNetV2((32, 32, 3),
                                              classes=10,
                                              weights=None)
    model.compile("adam",
                  "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client

    quality_vector = np.random.rand(3)
    cost_parameter = np.random.random_sample()
    # Start Flower client
    # HiddenPrints represses all Client Prints in the main terminal
    with HiddenPrints():
        fl.client.start_numpy_client(
            "localhost:8080",
            client=fC.FMoreClient(quality_vector=quality_vector,
                                  data=(x_train, y_train, x_test, y_test),
                                  model=model,
                                  cost_parameter=cost_parameter))


def run_simulation(num_rounds: int, num_clients: int):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(target=start_server,
                             args=(num_rounds, num_clients))
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(5)

    # Load the dataset partitions
    partitions = load(num_partitions=num_clients)

    # Start all the clients
    for partition in partitions:
        client_process = Process(target=start_client, args=(partition, ))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(num_rounds=2, num_clients=5)
