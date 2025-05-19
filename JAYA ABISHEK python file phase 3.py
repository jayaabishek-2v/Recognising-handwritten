# Handwritten Digit Recognition with Deep Learning
# Using TensorFlow/Keras and MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN input (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train and evaluate the model
def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model()
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining the model...")
    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=128, 
                        validation_split=0.2)
    
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Make predictions on some test samples
    print("\nMaking predictions on sample images...")
    sample_images = X_test[:5]
    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Display sample predictions
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return model, history

# Main execution
if __name__ == "__main__":
    print("=== Handwritten Digit Recognition with Deep Learning ===")
    model, history = train_and_evaluate()