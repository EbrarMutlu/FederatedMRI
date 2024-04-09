import flwr as fl
import keras as ks
from flwr.server import ServerConfig

from utils import load_testing_data

DEFAULT_SERVER_ADDRESS = "[::]:8080"
IMG_SIZE = 160

# Load testing data
X_test, y_test = load_testing_data()


# Define evaluation function
def evaluate(weights):
    model = create_model()  # Create a new model instance
    model.set_weights(weights)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("****** CENTRALIZED ACCURACY: ", accuracy, " ******")
    return loss, accuracy


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


# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,  # what percentage of clients we sample from in the next round
    min_available_clients=4,  # wait for 4 clients to connect before starting
)


# Define function to perform evaluation after each round
def evaluate_after_rounds(num_rounds):
    weights = None
    for round_num in range(num_rounds):
        # Perform federated averaging
        weights = strategy.aggregate(weights)

        # Perform evaluation
        loss, accuracy = evaluate(weights)
        print(f"Round {round_num + 1}: Loss = {loss}, Accuracy = {accuracy}")


if __name__ == '__main__':
    # Define the config with num_rounds
    config = ServerConfig(num_rounds=10)

    # Start the server with the correct parameters
    fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=config)
