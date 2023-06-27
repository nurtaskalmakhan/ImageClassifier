pip install tqdm

import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale = 1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (150, 150)


def load_data() :
  DIRECTORY = 
  CATEGORY = ["seg_train", "seg_test"]

  output = []

  for category in CATEGORY:
    path = os.path.join(DIRECTORY, category)
    print(path)
    images = []
    labels = []

    print("Loading {}".format(category))

    for folder in os.listdir(path):
      label = class_names_label[folder]

      for file in os.listdir(os.path.join(path, folder)):

        img_path = os.path.join(os.path.join(path, folder), file)

        image = cv2.imreaad(img_path)
        image = cv2.cvtCOlor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype='int32')

    output.append((images, labels))  

  return output



(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

def display_examples(class_names, images, labels):
  figsize = (20,20)
  fig = plt.figure(figsize=figsize)
  fig.suptitle("Some examples of images of the dataset", fontsize=16)
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = cv2.resize(image[i], figsize)
    plt.imshow(image.astype(np.uint8))
    plt.xlabel(class_names[labels[i]])
  plt.show()
display_examples(class_names, train_images, train_labels)

model = tf.keras.Sequental([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_images, train_labels, batch_size=100, epochs = 4, validation_split = 0.2)


def plot_accuracy_loss(history)
   fig = plt.figure(figsize=(10,5))
   
   plt.subplot(221)
   plt.plot(history.history['accuracy'],'bo--',label = "acc")
   plt.plot(history.history['val_accuracy'],'ro--',label = "val_acc")
   plt.title("train_acc vs val_acc")
   plt.ylabel("loss")
   plt.xlabel("epochs")

   plt.legend()
   plt.show()

plot_accuracy_loss(history)

test_loss = model.evaluate(test.images, test_labels)

predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis = 1)
print(classification_report(test_labels, pred_labels))


##################################

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

model = VGG16(weights = 'imagenet', include_top=False)
model = Model(input-model.inputs, outputs=model.layers[-5].output)

train_features = model.predict(train_images)
test_features = model.predict(test_images)


####################################


from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, Flatten

model2 = VGG16(weights = 'imagenet', include_top=False)

input_shape = model2.layers[-4].get_input_shape_at(0)
layer_input = Input(shape = (9, 9, 512))

x = layer_input
for layer in mode2.layers[-4::1]:
   x = layer(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x =Flatten()(x)
x =Dense(100, activation='relu')(x)
x = Dense(6,activation='softmax')(x)


new_model = Model(layer_input, x)

new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = new_model.fit(train_features, train_labels, batch_size=128, epochs=10, validation_split = 0.2)