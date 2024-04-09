import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import keras as ks

from utils import load_training_data, load_testing_data, load_partition

# Split training and testing into an 80/20 split
X_train, X_val, y_train, y_val = load_training_data(0.2)

# merge training and validation (tensorflow will handle that for us)
X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# load test data
X_test, y_test = load_testing_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train[1].shape)

model = ks.Sequential([
    ks.layers.Flatten(input_shape=(160, 160)),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dense(4)
])

model.compile(
    optimizer='adam',
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

num_epochs = 100

history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)


print(test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(num_epochs)

plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Testing Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
#plt.show()
plt.savefig("ML.jpg")


num_rounds = range(21)
accs = [0.26, 0.28, 0.35, 0.38, 0.5, 0.52, 0.55, 0.58, 0.56, 0.60, 0.59, 0.61, 0.60, 0.59, 0.58, 0.60, 0.60, 0.59, 0.60, 0.62, 0.62]

plt.plot(num_rounds, accs, label="Centralized Accuracy")
plt.xlabel("Number of FL rounds")
plt.ylabel("Centralized Accuracy")
plt.title("Centralized Accuracy vs Number of FL rounds")
plt.grid()
plt.savefig("FL.jpg")
#plt.show()

# Split training and testing into an 80/20 split
X_train, X_val, y_train, y_val = load_partition(0)

# merge training and validation (tensorflow will handle that for us)
X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# load test data
X_test, y_test = load_testing_data()

model = ks.Sequential([
    ks.layers.Flatten(input_shape=(160, 160)),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dense(4)
])

model.compile(
    optimizer='adam',
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

num_epochs = 50

history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(num_epochs)

plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Testing Accuracy")
plt.legend(loc="lower right")
plt.title("ML Model trained with 1/4 of Training Data")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
#plt.show()
plt.savefig("quarter_training_ML.jpg")