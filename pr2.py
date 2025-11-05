# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 23:19:02 2025

@author: monas
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 2D → 1D
    keras.layers.Dense(128, activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='softmax') # output layer
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss=%.3f" % test_loss)
print("Accuracy=%.3f" % test_acc)

train_loss, train_acc = model.evaluate(x_train, y_train)
print("Loss=%.3f" % train_loss)
print("Accuracy=%.3f" % train_acc)

n = random.randint(0, 9999)
plt.imshow(x_test[n])
plt.show()

predicted_value = model.predict(x_test)
print("The image is of type =", np.argmax(predicted_value[n]))

class_labels = [
 "T-shirt/Top","Trouser","Pullover","Dress","Coat",
 "Sandal","Shirt","Sneaker","Bag","Ankle Boot"
]
class_labels[np.argmax(predicted_value[n])]

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


