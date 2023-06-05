import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def main():
    """
    Initialize the Rice Leaf Health Classifier.
    """

    # An epoch is the full iteration on a dataset
    # There is no hard rules, but early stopping is essential
    #   especially when reaching desired accuracy or
    #   when it is deteriorating.
    # An example value would be 100/200, but must be changed
    #   based on the dataset, HW resources, actual dataset, etc.
    epochs = 200
    classes = 2  # healthy or infected
    shape = (50, 50, 3)  # Target Shape: 50x50 RGB images (JPEG)

    dataset_path = "datasets/labels.csv"
    features, _, labels = preprocess_dataset(dataset_path, classes, shape)
    dataset_size = len(features)
    
    labels = to_categorical(np.array(labels), num_classes=classes)

    # format model filepath
    model_filepath = f"models/{dataset_size}_{epochs}.h5"

    model = load_model_from_file(model_filepath, dataset_size, epochs=epochs)
    
    if model is None:
        model = train_new_model(
        model_filepath,
        features,
        labels,
        dataset_size,
        classes,
        shape,
        epochs=epochs)

    predict_new_samples(model, classes, shape)


def preprocess_dataset(dataset_path, classes, shape):
    """
    Load image data from the filenames in the CSV dataset.
    """

    # Load the CSV file using pandas
    datasets = pd.read_csv(dataset_path)

    # Prepare the dataset
    filenames = datasets["filename"].tolist()
    labels = datasets["label"].tolist()

    images = []
    for label, filename in zip(labels, filenames):
        filepath = f"datasets/samples/{filename}.jpg"
        if filename[0] != "X":
            filepath = f"datasets/healthy/{filename}.jpg"
            if label:
                filepath = f"datasets/infected/{filename}.jpg"
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=shape[:2])
        image = tf.keras.preprocessing.image.img_to_array(image)

        # Normalize the BGR pixels (0-1)
        image = image / 255.0
        images.append(image)

    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, filenames, labels


def load_model_from_file(model_filepath, dataset_size, epochs):
    """
    Load model from saved Hierarchical Data Format version 5 (H5) file.
    Initialize, compile, train, and save if not yet available.
    """

    print("\nLoading model")

    if os.path.exists(model_filepath):
        return load_model(model_filepath)

def train_new_model(
    model_filepath,
    features,
    labels,
    dataset_size,
    classes,
    shape,
    epochs=100):
    
    # Create the Sequential model
    model = Sequential()
    
    # In a Convolutional Layer, a NxM kernel slides to each possible location in the 2D input.
    # The kernel (of weights) performs an element-wise multiplication and summing all values into one.
    # The N-kernels will generate N-maps with unique features extracted from the input image.
    # Kernel Sizes: 3x3 (common), 5x5 (suitable for small features), 7x7 or 9x9 (appropriate for larger features)

    # Rectified Linear Unit ReLU(x) = max(0, x)
    # Any negative value becomes zero, addressing the gradients/derivatives
    #   from becoming very small and providing less effective learning
    # ReLU sets all negative values in the feature maps to zero
    #   introducing non-linearity to help in learning complex patterns and relationships
    
    # Batch Normalization helps accelerate and stabilize the training process
    #   by normalizing the activation after the Convolutional Layer.
    # Each feature map is independently normalized.
    
    # Max Pooling is used to downsample and reducing spatial dimensions of feature maps.
    # It divides the feature map into non-overlapping regions and chooses the maximum value for each.
    # Simply, it looks for the most important parts and reduces the data size for improved processing.
    # Larger pool size may lead to less detailed feature maps.
    # Pool Size: 2x2 and 3x3 (most common)
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # The complex feature maps must be flattened
    #   before feeding to the Dense layers, since it only accepts 1D arrays.
    model.add(Flatten())
    
    # Dense is a layer where each neuron is fully connected to the previous layer.
    # It means that each neuron accepts the full output of the previous layer.

    # Dropout is typically applied after the fully connected layers.
    # Value range from 0.2-0.5, with 0.5 as ideal to avoid overfitting in smaller datasets.

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output Layer
    # Sigmoid and SoftMax are applicable, but sigmoid is prefered for binary classifications.
    model.add(Dense(classes, activation="sigmoid"))

    # Compile the model
    # AdaM / Adaptive Moment Estimation
    # AdaM tends to reach an optimal solution faster.
    # Binary Cross-Entropy is more optimal for Binary Classification
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    model.fit(
        features,
        labels,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
    )

    # Save model to file
    model.save(model_filepath)

    return model


def predict_new_samples(model, classes, shape):
    """
    Batch predict the image dataset.
    """

    print("\nPredicting sample images.")
    dataset_path = "datasets/samples.csv"
    images, filenames, labels = preprocess_dataset(dataset_path, classes, shape)

    predictions = model.predict(images)

    # Leaf Health Classes
    leaf_results = ["Healthy", "Infected"]

    score = 0
    total = len(filenames)
    for filename, label, prediction in zip(filenames, labels, predictions):
        # Convert prediction to integer value
        
        pct = np.max(prediction) * 100
        index = np.argmax(prediction)
        result = leaf_results[index]

        # Check if actual label is the same with the prediction
        if label == index:
            score += 1
            print(f" / {filename}.jpg =\t{result} {pct:.0f}%")
            continue
        print(f" x {filename}.jpg =\t{result} {pct:.0f}%")

    # Display sample validation results
    print("---")
    accuracy = (score / total) * 100
    print(f"Accuracy: {score}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
