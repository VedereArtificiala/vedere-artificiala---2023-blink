import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set the paths for train and test data folders
train_dir = os.path.join("data", "asl_alphabet_train")
test_dir = os.path.join("data", "asl_alphabet_test")

# Set the image size to resize the images to
img_size = (64, 64)

# Function to convert a directory of image files to a NumPy array
def dir_to_array(directory):
    letter_dirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    img_array = []
    label_array = []

    for i, letter_dir in enumerate(letter_dirs):
        letter_path = os.path.join(directory, letter_dir)
        file_list = os.listdir(letter_path)
        for filename in file_list:
            img = Image.open(os.path.join(letter_path, filename)).convert('L')
            img = img.resize(img_size)
            img = np.array(img) / 255.0
            img_array.append(img)
            label = i
            label_array.append(label)

    return np.array(img_array), np.array(label_array)

def process_images(directory):
    file_list = os.listdir(directory)
    img_array = []
    label_array = []

    for filename in file_list:
        img = Image.open(os.path.join(directory, filename)).convert('L')
        img = img.resize(img_size)
        img = np.array(img) / 255.0
        img_array.append(img)
        label = ord(os.path.splitext(filename)[0]) - ord('A')
        label_array.append(label)

    return np.array(img_array), np.array(label_array)

# Preprocess the train data
train_data, train_labels = dir_to_array(train_dir)

# Split the data into train and validation sets
split_index = int(0.8 * len(train_data))
x_train, x_val = train_data[:split_index], train_data[split_index:]
y_train, y_val = train_labels[:split_index], train_labels[split_index:]

# Reshape the x_train array to have a single channel dimension
x_train = x_train.reshape(-1, 64, 64, 1)

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

# Visualize a few sample images from the training data
import matplotlib.pyplot as plt
import random

# Select random images from the dataset
num_images = 10
random_indices = random.sample(range(len(train_data)), num_images)

# Plot the selected images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i, idx in enumerate(random_indices):
    image = train_data[idx]
    label = train_labels[idx]

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20

# Train the model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) // 32,
          epochs=epochs,
          validation_data=(x_val, y_val))

# Preprocess the test data
test_data, test_labels = process_images(test_dir)

# Check the input shape
print("Test Data Shape:", test_data.shape)

# Normalize the test data
test_data = test_data / 255.0

test_data = test_data.reshape(-1, img_size[0], img_size[1], 1)

# Use the trained model to predict the letters in the test images
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the model's predictions
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='weighted')
recall = recall_score(test_labels, predicted_labels, average='weighted')
f1 = f1_score(test_labels, predicted_labels, average='weighted')

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
