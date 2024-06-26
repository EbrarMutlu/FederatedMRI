import os
import sys

import numpy as np
from flwr.client import start_numpy_client, NumPyClient
import keras as ks

from utils import load_partition
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod



# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMG_SIZE = 160
DEFAULT_SERVER_ADDRESS = "[::]:8080"

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

if len(sys.argv) > 1:
    X_train, X_val, y_train, y_val = load_partition(int(sys.argv[1]))
else:
    print("Not enough arguments... expecting python3 client.py PARTITION_NUMBER; where partition number is 0, 1, 2, 3")
    sys.exit()

class FederatedClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, steps_per_epoch=5, validation_split=0.1)

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return model.get_weights(), len(X_train), results

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val)
        print("****** CLIENT ACCURACY: ", accuracy, " ******")
        # Predict labels for validation data
        y_pred = np.argmax(model.predict(X_val), axis=1)

        # Calculate the number of correct guesses for each label
        correct_guesses = [np.sum((y_pred == i) & (y_val == i)) for i in range(4)]

        print("Correct Guesses for Each Label:", correct_guesses)

        return loss, len(X_val), {"accuracy": accuracy}


    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        trainloader, testloader = load_partition(partition_id=int(cid))
        return FederatedClient(trainloader, testloader).to_client()

    # Flower ClientApp
    app = ClientApp(
        client_fn=client_fn,
        mods=[
        
            fixedclipping_mod,
        ],
    )

if __name__ == '__main__':
    start_numpy_client(server_address="localhost:8080", client=FederatedClient())
