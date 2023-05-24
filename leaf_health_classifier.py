import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

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
    epochs = 100
    classes = 2 # healthy or infected
    shape = (50, 50, 3) # Target Shape: 50x50 RGB images (JPEG)
    
    dataset_path = "datasets/labels.csv"
    images, _, labels = preprocess_dataset(dataset_path, shape)
    dataset_size = len(images)

    features1, features2, target1, target2 = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    model = load_model_from_file(features1, target1, features2, target2, dataset_size, classes, shape, epochs=epochs, batch_size=32)
    
    predict_new_samples(model, shape)

def preprocess_dataset(dataset_path, shape):
    """
        Load image data from the filenames in the CSV dataset.
    """
    
    # Load the CSV file using pandas
    datasets = pd.read_csv(dataset_path)

    # Prepare the dataset
    filenames = datasets["filename"].tolist()
    labels = datasets["label"].tolist()
    
    augmented_labels = []
    
    # Rotations: 90, 180, 270
    rotations = [
        cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
    ]

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
        image = image/255.0
        images.append(image)
        augmented_labels.append(label)

        if filename[0] != "X":
            for rotation in rotations:
                images.append(cv2.rotate(image, rotation))
                augmented_labels.append(label)

    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(augmented_labels)
    
    return images, filenames, labels

def load_model_from_file(features1, target1, features2, target2, dataset_size, classes, shape, epochs=100, batch_size=32):
    """
        Load model from saved Hierarchical Data Format version 5 (H5) file.
        Initialize, compile, train, and save if not yet available.
    """
    
    print("\nLoading model")
    
    # format model filepath
    model_filepath = f"models/{dataset_size}_{epochs}.h5"

    if os.path.exists(model_filepath):
        model = load_model(model_filepath)
    else:
        # Create the Sequential model
        model = Sequential()

        # Add convolutional layers
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Flatten the output from the previous layer
        model.add(Flatten())

        # Add fully connected layers
        model.add(Dense(64, activation="relu"))

        model.add(Dense(128, activation="relu"))
        
        model.add(Dense(classes, activation="sigmoid"))

        # Compile the model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        print(model.summary())

        # Train the model
        model.fit(features1, target1, epochs=epochs, batch_size=batch_size, validation_data=(features2, target2))
        
        # Save model to file
        model.save(model_filepath)

    # Evaluate the model
    loss, accuracy = model.evaluate(features2, target2)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    return model

def predict_new_samples(model, shape):
    """
        Batch predict the image dataset.
    """

    print("\nPredicting sample images.")
    dataset_path = "datasets/samples.csv"
    images, filenames, labels = preprocess_dataset(dataset_path, shape)
    
    predictions = model.predict(images)
    
    # Leaf Health Classes
    leaf_results = ["Healthy", "Infected"]
    
    score = 0
    total = len(filenames)
    for filename, label, prediction in zip(filenames, labels, predictions):
        # Convert prediction to integer value
        prediction_value = int(round(prediction[0]))
        result = leaf_results[prediction_value]
        
        # Check if actual label is the same with the prediction
        if label == prediction_value:
            score += 1
            print(f" / {filename}.jpg =\t{result}")
            continue
        print(f" x {filename}.jpg =\t{result}")

    # Display sample validation results
    print("---")
    accuracy = (score/total)*100
    print(f"Accuracy: {score}/{total} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()