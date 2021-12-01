# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FMore (FedAuction) [Rongfei Zeng et al., 2020] strategy.

Paper: https://arxiv.org/abs/2002.09699
"""
import numpy as np
import inspect

from logging import WARNING, INFO, DEBUG
from typing import Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""


class FedAuction(Strategy):
    def __init__(
        self,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        scoring_function=lambda q: np.sum(q)
    ) -> None:
        """Federated Auctioning strategy.

        Implementation based on https://arxiv.org/abs/2002.09699

        Parameters
        ----------
        min_available_clients : int, optional
            Least number of clients for federated learning to start, by default 2
        min_fit_clients : int, optional
            [description], by default 2
        accept_failures : bool, optional
            Boolean whether the server should accept failures or not, by default True
        initial_parameters : Optional[Parameters], optional
            A way to provide model parameter prior to the learning process, by default None
        scoring_function : [type], optional
            Function which determines the rules for bid calculation in clients, by default lambdaq:np.sum(q)
        """

        super().__init__()
        self.min_available_clients = min_available_clients
        self.timeout = 86400
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        self.scoring_function = scoring_function

    def __repr__(self) -> str:
        rep = f"FedAuction(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(
                weights=initial_parameters)
        return initial_parameters

    def evaluate(
            self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        # Remark:: implement different evaluate if needed
        return None

    def bid_ask(self, client_manager: ClientManager):
        """Collects the bids from every available client.

        Parameters
        ----------
        parameters : Parameters
            Modelparameters for a client.
        client_manager : ClientManager
            The client manager through which the clients can be reached.

        Returns
        -------
        list(tuple(list(float), float, ClientProxy))
            List of bids with one bid like tuple(quality_vector, payment, node).
        """
        # Fit_instructions
        config = {}
        # Split function at '=' for easier use
        config["scoring"] = "".join(
            inspect.getsource(self.scoring_function).split("=")[1:])

        # All client information as list
        clients = client_manager.all()

        # Retrieve bids/scores from each client and store them in dictonary with score as key
        collected_bids = []
        for key in clients.keys():
            # If client doesnt have a score ignore it
            metrics = clients[key].evaluate(
                FitIns(Parameters(None, None), config)).metrics
            if "payment" in metrics:
                # Split string and extract float values
                collected_bids.append(
                    ([float(x) for x in metrics["quality_vector"].split(",")],
                     metrics["payment"], clients[key]))

        return collected_bids

    def winner_determination(self, bids):
        """Calculates score and determines Node with highest score.

        Parameters
        ----------
        bids : list(tuple(list(float),float,ClientProxy))
            List of bids with one bid like tuple(quality_vector,payment,node).

        Returns
        -------
        ClientProxy of Node with highest score.
        
        Notes
        -----
        Currently returns the first encountered Client with the highest score.
        A different selection strategy in the case of multiple highest bids can be implemented.
        """

        scores = [(self.scoring_function(q) - p, c) for q, p, c in bids]

        winner = max(scores, key=lambda b: b[0])[1]
        return winner

    def configure_fit(
            self, rnd: int, parameters: Parameters,
            client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
            
        Returns
        -------
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
        is not included in this list, it means that this `ClientProxy`
        will not participate in the next round of federated learning.
        """

        # Wait for min_available_clients for FL to start
        if not client_manager.wait_for(num_clients=self.min_available_clients,
                                       timeout=self.timeout):
            # Do not continue if not enough clients are available
            log(INFO, "FMore: not enough clients available after timeout %s",
                self.timeout)
            return []
        else:
            # Continue if enough clients are available
            log(
                INFO,
                "FMore: enough clients available for FMore to start training round %s",
                rnd)

        # Bid Ask and Bid Collect
        bids = self.bid_ask(client_manager)

        # Log all values for debugging purpose
        bids.sort(key=lambda b: self.scoring_function(b[0]) - b[1],
                  reverse=True)
        for i, bid in enumerate(bids):
            log(
                DEBUG,
                f"Rank {i + 1} - Client {bid[2]}: Quality_Vector {bid[0]}, Payment {bid[1]}, Utility {self.scoring_function(bid[0])}, Score {self.scoring_function(bid[0]) - bid[1]}"
            )

        # Winner Determination
        winner = self.winner_determination(bids)

        # Pay Client that was chosen to train
        self.pay_client(winner)

        # Return client/config pairs
        return [(winner, FitIns(parameters, {}))]

    def pay_client(self, client):
        """Pay Client.
        
        Arguments
        ---------
        client: ClientProxy
            The client that needs to get payed.
            
        Returns
        -------
        Either True when payment is successful or False when payment is unsuccessful.
        """

        # retrieve Client payment details
        config = {"payment_details": True}
        metrics = client.evaluate(FitIns(Parameters(None, None),
                                         config)).metrics
        if "payment_details" in metrics:
            # Start payment...
            # End Payment
            return True
        return False

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.
        
        Arguments
        ---------
        rnd: Integer
            The current round of federated learning.
        parameters: Parameters
            The current (global) model parameters.
        client_manager: ClientManager
            The client manager which holds all currently connected clients.
            
        Returns
        -------
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `EvaluateIns` for this particular `ClientProxy`. If a particular
        `ClientProxy` is not included in this list, it means that this
        `ClientProxy` will not participate in the next round of federated
        evaluation.
        """

        # Remark:: implement different configure_evaluate if needed
        # Decides on which clients to evaluate the current global model
        # Goes on to evalute the model with data present on chosen Clients

        # Parameters and config
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # All clients
        clients = list(client_manager.all().values())

        # Return all client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.
        
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the server was waiting for client
            updates.
            
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [(parameters_to_weights(fit_res.parameters),
                            fit_res.num_examples)
                           for client, fit_res in results]
        return weights_to_parameters(aggregate(weights_results)), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.
        
        Parameters
        ----------
        rnd: int
            The current round of federated learning.
        results: List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured clients. 
            Each pair of `(ClientProxy, FitRes)` constitutes a successful update from one of the
            previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures: List[BaseException]
            Exceptions that occurred while the server was waiting for client updates.
                
        Returns
        -------
        Optional `float` representing the aggregated evaluation result. Aggregation
        typically uses some variant of a weighted average.
        """

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg([(
            evaluate_res.num_examples,
            evaluate_res.loss,
            evaluate_res.accuracy,
        ) for _, evaluate_res in results])

        loss_average = np.mean(
            [evaluate_res.loss for _, evaluate_res in results])
        accuracy_average = np.mean(
            [evaluate_res.metrics["accuracy"] for _, evaluate_res in results])

        log(
            DEBUG,
            f"Average loss: {loss_average}, Average accuracy: {accuracy_average}"
        )

        return loss_aggregated, {}
