# @title v4
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from google.colab import files

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualizza un paio di immagini di training con le loro etichette
print('Visualizzo un paio di immagini di training, con le loro etichette')
print()
plt.figure(figsize=(8, 4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.title(f"Etichetta: {y_train[i]}")
    plt.axis('on')
plt.show()
print('')

# Normalizza le immagini
x_train = x_train / 255.0
x_test = x_test / 255.0

# Regolarizzazione: Per evitare overfitting
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello e salva la storia
print('Addestramento modello')
print('')
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Valuta il modello
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('')
print(f'Accuratezza sul set di test: {accuracy:.4f}')

# Plotta l'accuratezza
print('')
print('Grafico accuratezza')
print()
plt.figure(facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')
plt.plot(history.history['accuracy'], label='Accuratezza training', color='cyan')
plt.plot(history.history['val_accuracy'], label='Accuratezza validation', color='magenta')
plt.title('Accuratezza durante l\'addestramento', color='white')
plt.xlabel('Epoca', color='white')
plt.ylabel('Accuratezza', color='white')
plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
plt.tick_params(colors='white')
plt.show()

# Plotta i pesi del primo layer Dense
print('')
print('Pesi del primo layer Dense')
print('')
weights_layer1 = model.layers[1].get_weights()[0]
n_neurons_layer1 = weights_layer1.shape[1]
n_cols_layer1 = 16
n_rows_layer1 = (n_neurons_layer1 + n_cols_layer1 - 1) // n_cols_layer1

plt.figure(figsize=(n_cols_layer1 * 1.5, n_rows_layer1 * 1.5), facecolor='black')
for i in range(n_neurons_layer1):
    plt.subplot(n_rows_layer1, n_cols_layer1, i + 1)
    weight_image = weights_layer1[:, i].reshape(28, 28)
    plt.imshow(weight_image, cmap='gray')
    plt.title(f'Neurone {i}', color='white')
    plt.axis('off')
plt.suptitle('Pesi dei 128 neuroni del primo layer', color='white', fontsize=16)
plt.tight_layout()
plt.show()

# Plotta i pesi del secondo layer Dense
print('')
print('Pesi del secondo layer Dense')
print('')
weights_layer2 = model.layers[3].get_weights()[0]
n_neurons_layer2 = weights_layer2.shape[1]
n_cols_layer2 = 5
n_rows_layer2 = (n_neurons_layer2 + n_cols_layer2 - 1) // n_cols_layer2

plt.figure(figsize=(n_cols_layer2 * 2, n_rows_layer2 * 2), facecolor='black')
for i in range(n_neurons_layer2):
    plt.subplot(n_rows_layer2, n_cols_layer2, i + 1)
    weight_vector = weights_layer2[:, i]
    plt.plot(weight_vector, color='cyan')
    plt.title(f'Cifra {i}', color='white')
    plt.axis('off')
plt.suptitle('Pesi dei 10 neuroni del secondo layer', color='white', fontsize=16)
plt.tight_layout()
plt.show()
print('')

# Fai previsioni sulle prime 5 immagini del test set
print('Previsioni sulle prime 5 immagini del test set:')
print('')
predictions = model.predict(x_test[:5])
for i in range(5):
    predicted_label = tf.argmax(predictions[i]).numpy()
    true_label = y_test[i]
    print(f"Immagine {i}: Predetta = {predicted_label}, Reale = {true_label}")

# Funzione per caricare e predire un'immagine
def predict_digit(loop=True, invert=True):
    while True:
        print()
        print('Caricamento immagine')
        print()
        try:
            uploaded = files.upload()
            for filename in uploaded.keys():
                img = Image.open(filename).convert('L')
                # Ridimensiona l'immagine usando BOX
                img = img.resize((28, 28), Image.Resampling.BOX)
                img_array = np.array(img)
                # Binarizza con inversione opzionale
                if invert:
                    img_array = (img_array < 128).astype(np.float32)  # Per sfondo nero, testo bianco
                else:
                    img_array = (img_array > 127).astype(np.float32)  # Per sfondo bianco, testo nero

                plt.imshow(img_array, cmap='gray')
                plt.title("Immagine originale", color='white')
                plt.axis('off')
                plt.show()

                img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
                img_array = tf.reshape(img_array, [1, 28, 28])

                prediction = model.predict(img_array)
                predicted_digit = tf.argmax(prediction[0]).numpy()
                probabilities = prediction[0]

                print("Probabilità per ogni cifra (0-9):")
                for i, prob in enumerate(probabilities):
                    print(f"Cifra {i}: {prob:.4f}")
                plt.imshow(img_array[0], cmap='gray')
                plt.title(f"Predetto: {predicted_digit}", color='white')
                plt.axis('off')
                plt.show()
                print(f"La cifra predetta è: {predicted_digit}")
        except Exception as e:
            print(f"Errore durante il caricamento: {e}")

        if not loop:
            break
        action = input("\nVuoi caricare un'altra immagine? (Sì/No o Exit): ").lower().strip()
        if len(action) > 5:
            print("Errore: L'input supera i 5 caratteri.")
            continue
        if action in ['no', 'exit']:
            print("Terminazione del programma.")
            break

# Esegui la funzione con loop e inversione
predict_digit(loop=True, invert=True)