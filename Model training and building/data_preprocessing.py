# Install necessary libraries
!pip install opencv-python-headless
!pip install scikit-learn
!pip install tensorflow
!pip install matplotlib
!pip install Pillow
!pip install kaggle
!kaggle datasets download -d ducnguyen168/dataset-skin-tone
!unzip dataset-skin-tone.zip

# data Pre-processing
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data directory and categories
data_dir = "data_skintone"
categories = ["dark", "light", "mid-dark", "mid-light"]
img_size = 64  # You can change this size based on your requirement

def create_dataset(data_dir, categories, img_size):
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass

    return np.array(data), np.array(labels)

data, labels = create_dataset(data_dir, categories, img_size)

# Normalize pixel values
data = data / 255.0

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(train_data)

