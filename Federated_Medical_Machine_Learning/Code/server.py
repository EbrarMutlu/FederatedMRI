from typing import Dict, Optional, Tuple, List

import flwr as fl
import keras as ks
from flwr.server import ServerConfig
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
        return loss, {"accuracy": accuracy}
    return evaluate




# Define strategy using the evaluation function
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,
    min_available_clients=4,
    evaluate_fn=get_eval_fn(),  # Ensure this function uses the correct 'evaluate' that utilizes 'create_model'

)

app = ServerApp()

if __name__ == '__main__':
    config = ServerConfig(num_rounds=10)
    fl.server.start_server(server_address=f"{server_address}:{port_number}", strategy=strategy, config=config)
