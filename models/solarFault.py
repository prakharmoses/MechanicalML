import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
# %matplotlib inline 

import tensorflow as tf
import random
from cv2 import resize
from glob import glob
import os
import subprocess

import warnings
import json

# from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
import visualkeras


warnings.filterwarnings("ignore")


# class CustomObjectEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, type(Ellipsis)):
#             return str(obj)
#         return super().default(obj)


if not os.path.exists('models/solar_panel_fault_detection.keras'):
    img_height = 244
    img_width = 244
    train_ds = tf.keras.utils.image_dataset_from_directory(
    './models/Faulty_solar_panel',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    './models/Faulty_solar_panel',
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True)


    class_names = train_ds.class_names
    # print("The class name is: ", class_names)
    # print("The train_ds is: ", train_ds)


    # plt.figure(figsize=(15, 15))
    # for images, labels in train_ds.take(1):
    #     for i in range(25):
    #         ax = plt.subplot(5, 5, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")


    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False


    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(90)(x)
    model = tf.keras.Model(inputs, outputs)

    # print("The model summary at stage 1 is:\n", model.summary())


    # plot_model(model, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)


    # visualkeras.layered_view(model,legend=True,spacing=50,background_fill = 'white')


    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


    epoch = 15
    model.fit(train_ds, validation_data=val_ds, epochs=epoch,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=1,
                restore_best_weights=True
            )
        ]
    )


    base_model.trainable = True
    for layer in base_model.layers[:14]:
        layer.trainable = False
    # print("The model summary at stage 2 is:\n", model.summary())


    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


    epoch = 15
    history = model.fit(train_ds, validation_data=val_ds, epochs=epoch,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=1,
            )
        ]
    )


    get_ac = history.history['accuracy']
    get_los = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs = range(len(get_ac))
    # plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
    # plt.plot(epochs, get_los, 'r', label='Loss of Training data')
    # plt.title('Training data accuracy and loss')
    # plt.legend(loc=0)
    # plt.figure()

    # plt.plot(epochs, get_ac, 'g', label='Accuracy of Training Data')
    # plt.plot(epochs, val_acc, 'r', label='Accuracy of Validation Data')
    # plt.title('Training and Validation Accuracy')
    # plt.legend(loc=0)
    # plt.figure()

    # plt.plot(epochs, get_los, 'g', label='Loss of Training Data')
    # plt.plot(epochs, val_loss, 'r', label='Loss of Validation Data')
    # plt.title('Training and Validation Loss')
    # plt.legend(loc=0)
    # plt.figure()
    # plt.show()


    loss, accuracy = model.evaluate(val_ds)

    # plt.figure(figsize=(20, 20))
    # for images, labels in val_ds.take(1):
    #     for i in range(16):
    #         ax = plt.subplot(4, 4, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         predictions = model.predict(tf.expand_dims(images[i], 0))
    #         score = tf.nn.softmax(predictions[0])
    #         if(class_names[labels[i]]==class_names[np.argmax(score)]):
    #             plt.title("Actual: "+class_names[labels[i]])
    #             plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'green'})
                
    #         else:
    #             plt.title("Actual: "+class_names[labels[i]])
    #             plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'red'})
    #         plt.gca().axes.yaxis.set_ticklabels([])        
    #         plt.gca().axes.xaxis.set_ticklabels([])

    model.save('models/solar_panel_fault_detection.keras')
else:
    # with open('models/solar_panel_fault_detection.h5', 'r') as f:
    #     model = tf.keras.models.load_model(f)
        # model_json = json.load(f, cls=CustomObjectEncoder)
    
    # model = tf.keras.models.model_from_json(model_json)
    model = tf.keras.models.load_model('models/solar_panel_fault_detection.keras')


























# # --------------------------------------------- Own code ---------------------------------------------
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from keras.optimizers import Adam
# from keras.metrics import Precision, Recall
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import numpy as np
# from PIL import Image  # For image processing
# from sklearn.model_selection import train_test_split
# import os

# # Clearing older preprocessed data
# if os.path.exists('data'):
#     for file in os.listdir('data'):
#         os.remove(os.path.join('data', file))
#     os.rmdir('data')

# # ------------------------- Preprocessing Data -------------------------
# # Define data path (replace with the actual path to your folder containing the six subfolders)
# data_path = 'Faulty_solar_panel'

# # Assuming all images are RGB and have the same size (replace with actual values)
# IMG_WIDTH, IMG_HEIGHT = 28, 28
# NUM_CHANNELS = 3

# # Define split ratios for train, validation, and test sets (adjust as needed)
# test_size = 0.1  # 20% for test data
# val_size = 0.2  # 10% for validation data
# train_size = 1 - test_size - val_size  # Remaining for training data

# # Define class labels from folder names
# class_labels = ['Clean', 'Bird-drop', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']


# def load_images_and_labels(folder_path, is_training=True):
#   """Loads images and labels from a folder."""
#   X, y = [], []
#   for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.jpg', '.jpeg')):  # Check image extensions
#       image_path = os.path.join(folder_path, filename)
      
#       # Extract label from folder name
#       label = folder_path.split('\\')[-1]  # Get parent folder name as label
      
#       # Append image and label
#       X.append(load_image(image_path))
#       y.append(class_labels.index(label))
  
#   # Split data into training and validation sets (if training)
#   if is_training:
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
#     return X_train, y_train, X_val, y_val
#   else:
#     return X, y  # Return all data for testing

  
# def load_image(image_path):
#   """Loads and preprocesses an image."""
# #   image = Image.open(image_path).convert('RGB')  # Convert to RGB mode if necessary
#   image = Image.open(image_path).convert('L')  # Convert to grayscale mode
#   image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize image if needed
#   return np.array(image)


# # Load and split data for each class
# all_data = {}
# for class_label in class_labels:
#   folder_path = os.path.join(data_path, class_label)
# #   print("The folder path is = ", folder_path)
  
#   # Load training, validation, and test data
#   if os.path.isdir(folder_path):  # Check if folder exists
#     X_train, y_train, X_val, y_val = load_images_and_labels(folder_path)
#     X_test, y_test = load_images_and_labels(folder_path, is_training=False)
#     all_data[class_label] = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
#   else:
#     print(f"Warning: Folder '{folder_path}' not found. Skipping class '{class_label}'.")

# if not os.path.exists('data'):
#     os.mkdir('data')
# os.chdir('data')


# # Save data as NumPy array files (replace with desired filenames)
# X_train, y_train = [], []
# X_val, y_val = [], []
# X_test, y_test = [], []

# for class_label, data in all_data.items():
#   X_train_here, y_train_here = data['train']
#   X_val_here, y_val_here = data['val']
#   X_test_here, y_test_here = data['test']

#   X_train.extend(X_train_here)
#   y_train.extend(y_train_here)
#   X_val.extend(X_val_here)
#   y_val.extend(y_val_here)
#   X_test.extend(X_test_here)
#   y_test.extend(y_test_here)
  
# #   np.save(f'train_images_{class_label}.npy', X_train)
# #   np.save(f'train_labels_{class_label}.npy', y_train)
# #   np.save(f'val_images_{class_label}.npy', X_val)
# #   np.save(f'val_labels_{class_label}.npy', y_val)
# #   np.save(f'test_images_{class_label}.npy', X_test)
# #   np.save(f'test_labels_{class_label}.npy', y_test)

# np.save(f'train_images.npy', X_train)
# np.save(f'train_labels.npy', y_train)
# np.save(f'val_images.npy', X_val)
# np.save(f'val_labels.npy', y_val)
# np.save(f'test_images.npy', X_test)
# np.save(f'test_labels.npy', y_test)

# print("Data successfully converted and saved for training, validation, and testing.")
# os.chdir('..')

# # ------------------------- Loading Data -------------------------
# X_train = np.load('./data/train_images.npy')
# X_val = np.load('./data/val_images.npy')
# X_test = np.load('./data/test_images.npy')

# y_train = np.squeeze(np.load('./data/train_labels.npy'))
# y_val = np.squeeze(np.load('./data/val_labels.npy'))
# y_test = np.squeeze(np.load('./data/test_labels.npy'))

# X_train = np.expand_dims(X_train,axis = 3)
# X_val = np.expand_dims(X_val,axis = 3)
# X_test = np.expand_dims(X_test,axis = 3)

# X_train = X_train / 255.0
# X_val = X_val / 255.0
# X_test = X_test / 255.0


# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
# history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
# print(history.history.keys())


# # Plotting the training and validation accuracy
# plt.plot(history.history['accuracy'], label = 'Accuracy')
# plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
# plt.plot(history.history['loss'], label = 'loss')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()