import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1)

# Extract data and labels
data = mnist.data
labels = mnist.target.astype('int')  # Convert labels to int for compatibility with TensorFlow

# Split the data into train and test sets (80-20 split)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape the data to 28x28x1 (grayscale image format)
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoded format (necessary for categorical classification)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define model parameters
input_shape = (28, 28, 1)
batch_size = 128
num_classes = 10
epochs = 10

# Build the CNN model using tf.keras.Sequential
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # Dropout to reduce overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model using RMSprop optimizer directly from tf.keras
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the trained model
model.save('mnist_digit_recognizer_rmsprop.h5')
print("Model saved as mnist_digit_recognizer_rmsprop.h5")
