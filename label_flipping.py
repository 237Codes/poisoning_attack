
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


# 1. Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# 2. Load Pre-trained ResNet50 (excluding top layer for custom classification)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# 3. Implement Label Flipping Attack
def label_flipping_attack(y, flip_rate):
    num_samples = y.shape[0]
    num_to_flip = int(num_samples * flip_rate)
    indices_to_flip = np.random.choice(num_samples, num_to_flip, replace=False)

    flipped_labels = np.copy(y)

    for i in indices_to_flip:
        # Flip the label to a random class (excluding the original)
        original_class = np.argmax(y[i])
        new_class = np.random.choice([j for j in range(10) if j != original_class])
        flipped_labels[i] = to_categorical(new_class, num_classes = 10)

    return flipped_labels

# 4. Implement Class Mislabeling Attack
def class_mislabeling_attack(y, target_class_from, target_class_to):
    mislabeled_y = np.copy(y)
    for i in range(len(y)):
        if np.argmax(y[i]) == target_class_from:
            mislabeled_y[i] = to_categorical(target_class_to, num_classes = 10)
    return mislabeled_y


# --- Model Training and Evaluation (with comparison) ---

# Define a function for training and evaluation
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, loss, accuracy


# Train the model without any attack
history_clean, loss_clean, accuracy_clean = train_and_evaluate(model, x_train, y_train, x_test, y_test)

# Train the model with the label flipping attack
y_train_flipped = label_flipping_attack(y_train, 0.2)
history_flipped, loss_flipped, accuracy_flipped = train_and_evaluate(tf.keras.models.clone_model(model), x_train, y_train_flipped, x_test, y_test)


# Plotting the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history_clean.history['val_accuracy'], label='Clean Data')
plt.plot(history_flipped.history['val_accuracy'], label='Flipped Labels')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy Comparison')

plt.subplot(1, 2, 2)
plt.bar(['Clean Data', 'Flipped Labels'], [accuracy_clean, accuracy_flipped])
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.show()

print(f"Clean Data Test Accuracy: {accuracy_clean}")
print(f"Flipped Labels Test Accuracy: {accuracy_flipped}")