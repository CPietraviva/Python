import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizza le immagini
x_train = x_train / 255.0
x_test = x_test / 255.0

# Definisci il modello
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Appiattisce l'immagine 28x28 in un vettore di 784 elementi
    Dense(128, activation='relu'), # Primo layer denso con 128 neuroni e funzione di attivazione ReLU
    Dense(10, activation='softmax') # Secondo layer denso (output) con 10 neuroni (cifre da 0 a 9) e softmax
])

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello
model.fit(x_train, y_train, epochs=5)

# Valuta il modello
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuratezza sul set di test: {accuracy:.4f}')