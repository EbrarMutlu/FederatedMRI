from typing import Dict, Optional, Tuple, List, Union

import flwr as fl
import keras as ks
import numpy as np
#from flwr.common import FitRes, Parameters, Scalar
#from flwr.server import ServerConfig
#from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

from utils import load_testing_data
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig


IMG_SIZE = 160

# Load server address and port number from command-line arguments or use default
server_address = "0.0.0.0"
port_number = "8080"


# Create model function
def create_model():
    model = ks.Sequential([
        ks.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
        ks.layers.Dense(128, activation='relu'),
        ks.layers.Dense(4)
    ])
    model.compile(
        optimizer='adam',
        loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model
##lass SaveModelStrategy(fl.server.strategy.FedAvg):
##   def aggregate_fit(
##       self,
##       server_round: int,
##       results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
##       failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
##   ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
##
##       # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
##       aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
##
##       if aggregated_parameters is not None:
##           # Convert `Parameters` to `List[np.ndarray]`
##           aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
##
##           # Save aggregated_ndarrays
##           print(f"Saving round {server_round} aggregated_ndarrays...")
##           np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)
##
##       return aggregated_parameters, aggregated_metrics
##

# Define the evaluation function
def get_eval_fn():
    def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        X_test, y_test = load_testing_data()
        model = create_model()
        # Correctly convert NDArrays (Flower parameters) to Keras-compatible weights
        keras_weights = fl.common.parameters_to_ndarrays(fl.common.ndarrays_to_parameters(weights))
        model.set_weights(keras_weights)
        loss, accuracy = model.evaluate(X_test, y_test)
        print("****** CENTRALIZED ACCURACY: ", accuracy, " ******")
        # Predict labels for testing data
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Calculate the number of correct guesses for each label
        correct_guesses = [np.sum((y_pred == i) & (y_test == i)) for i in range(4)]

        print("Correct Guesses for Each Label:", correct_guesses)


        return loss, {"accuracy": accuracy}

    
    return evaluate


# Define strategy using the evaluation function
# Define the initial strategy
initial_strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,
    min_available_clients=4,
    evaluate_fn=get_eval_fn(),  # Ensure this function uses the correct 'evaluate' that utilizes 'create_model'
)


strategy = DifferentialPrivacyClientSideFixedClipping(
    initial_strategy, noise_multiplier=0.15, clipping_norm=2, num_sampled_clients=4
)

app = ServerApp()

if __name__ == '__main__':
    config = ServerConfig(num_rounds=10)
    fl.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")
    fl.server.start_server(server_address=f"{server_address}:{port_number}", strategy=strategy, config=config)

