# -*- coding: utf-8 -*-
"""African Skin Disease Classification — original Google Colab export.

Historical record of the notebook used during the master's research.
Contains Colab-only constructs (!pip, drive.mount, shell magics) and
hard-coded Google Drive paths; it will not run as a plain local script.

Prefer train_classifier.py and preprocess_heic.py for reproducible use.

Original Colab notebook:
https://colab.research.google.com/drive/1hjM6sEcHm0X3TD4os4-N99Wt-0CovVVE

# Preprocessing

## ** PROBLEM STATEMENT **

Image gotten to trail my ML model weere snapped with an iPhone encoded with .HEIC format which is not readable, this task is to convert them to JPG format!
"""

!cd /usr/src/

!git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.0

cd ImageMagick-7.1.0/

!./configure --with-modules

!make

!sudo make install

!convert

!sudo ldconfig /usr/local/lib

!convert

!pip install heic-to-jpg

import os
from google.colab import drive
from PIL import Image

drive.mount('/content/drive/')

# Get a list of all files in the Skin diseases folder.
files = os.listdir('/content/drive/My Drive/Skin diseases/data')

print(len(files))

!heic-to-jpg -s "/content/drive/My Drive/Skin diseases/child.HEIC" --keep



# Iterate over the files and convert the HEIC files to JPEG.
for file in files:
  if file.endswith('.HEIC'):
    # Get the file path.
    file_path = f"/content/drive/My Drive/Skin diseases/{file}"
    print(file_path)
    # Convert the file to JPEG.
    !heic-to-jpg -s "/content/drive/My Drive/Skin diseases/{file}"

"""Model development"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Define constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 64
EPOCHS = 20
# files = os.listdir('/content/drive/My Drive/Skin diseases')
NUM_CLASSES = len(files)
DATA_DIR = '/content/drive/My Drive/Skin diseases/data'  # Path to directory containing all images

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.6,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Increasing rotation range
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=60,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.3,
#     zoom_range=0.3,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
# )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(150, 150),  # Set the target size of the images
    batch_size=BATCH_SIZE,                # Set the batch size
    class_mode='categorical',             # Set the class mode
    shuffle=False                         # Don't shuffle the data
)

# Get a batch of images and labels from the generator
images, labels = train_generator.next()

# Plot a few images
import matplotlib.pyplot as plt

# Define the class labels
class_labels = list(train_generator.class_indices.keys())

# Plot the images with their corresponding labels
plt.figure(figsize=(10, 10))
for i in range(9):  # Plot 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(class_labels[labels[i].argmax()])
    plt.axis('off')
plt.show()

# Load and augment data
image_files = []
labels = []

!ls

for class_folder in os.listdir('/content/drive/My Drive/Skin diseases/data'):
    class_folder_path = os.path.join('/content/drive/My Drive/Skin diseases/data', class_folder)
    if os.path.isdir(class_folder_path):
        for file in os.listdir(class_folder_path):
            if file.endswith(('jpg', 'jpeg')):
                image_files.append(os.path.join(class_folder_path, file))
                labels.append(class_folder)

# Split data into train and validation sets
train_files, validation_files, train_labels, validation_labels = train_test_split(image_files, labels, test_size=0.25, random_state=42)

# Create DataFrame
train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
validation_df = pd.DataFrame({'filename': validation_files, 'class': validation_labels})

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_dataframe(
    validation_df,
    x_col='filename',
    y_col='class',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')  # Number of classes
])

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Dropout(0.25),  # Add dropout layer
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),  # Add dropout layer
#     Dense(NUM_CLASSES, activation='softmax')
# ])

# Load the pre-trained VGG16 model without the top (classification) layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers of the pre-trained model so they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the pre-trained model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=output)

# Define a custom optimizer
custom_optimizer = Adam(learning_rate=0.0001)

# Compile
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# Define a learning rate schedule function
def learning_rate_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    decay_rate = 0.1    # Decay rate

    if epoch < 10:
        return initial_lr
    elif 10 <= epoch < 20:
        return initial_lr * decay_rate
    else:
        return initial_lr * (decay_rate ** 2)

# Create a LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(learning_rate_schedule)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    callbacks=[lr_scheduler]
)

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     verbose=1
# )

# model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=len(validation_generator))

# Plot the learning rate schedule
plt.plot(history.epoch, [learning_rate_schedule(epoch) for epoch in history.epoch])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()

# Calculate the total number of images used in training
total_images = len(train_generator) * train_generator.batch_size

print("Total number of images used in training:", total_images)

total_images = len(validation_generator) * validation_generator.batch_size

print("Total number of images used in validating:", total_images)

# Evaluate model
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy}')

# Confusion Matrix
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validation_generator.classes, y_pred)
print('Confusion Matrix:')
print(cm)

from sklearn.metrics import f1_score

# Calculate micro F1 score
micro_f1 = f1_score(y_true, y_pred, average='micro')

# Calculate macro F1 score
macro_f1 = f1_score(y_true, y_pred, average='macro')

print("Micro F1 Score:", micro_f1)
print("Macro F1 Score:", macro_f1)

# Classification Report
target_names = list(validation_generator.class_indices.keys())
print('Classification Report:')
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# ROC Curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(validation_generator.classes, Y_pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label=f'ROC curve (area = {roc_auc[0]:0.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

