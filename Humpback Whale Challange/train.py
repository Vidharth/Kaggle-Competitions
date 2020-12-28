import os
import time
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from numba import cuda
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


cuda.select_device(0)
cuda.close()
K.clear_session()
gc.collect()

K.tensorflow_backend._get_available_gpus()

K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(
    log_device_placement=True, allow_soft_placement=True, 
    gpu_options=tf.GPUOptions(allow_growth=True), device_count={'GPU': 10, 'CPU': 10})))

plt.style.use('ggplot')

train_data = pd.read_csv("train.csv")

print(len(train_data["Image"]))
images = np.zeros((len(train_data["Image"]), 64, 64, 3))
print(images.shape)
x = 0
for img in (train_data["Image"]):
    pic = image.load_img(
        "D:/STUDY/PROJECTS/humpback-whale-identification/train/"+img, target_size=(64, 64, 3))
    n = image.img_to_array(pic)
    n = preprocess_input(n)
    images[x] = n/255.0
    if (x % 500 == 0):
        print("Pre Processing Image: " + str(x) + "," + str(img))
    x += 1
print(images.shape)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(train_data["Id"])
print(Y.shape)
Y = to_categorical(Y)
print(Y.shape)

del train_data
""" train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, width_shift_range=0.1, 
     height_shift_range=0.1, shear_range=0.1, 
     zoom_range=0.1)
train_datagen.fit(images)
X = train_datagen.flow(x, y=None, batch_size=64)
print(X.shape)
 """
classifier = Sequential()

classifier.add(Conv2D(128, (5, 5), input_shape=(64, 64, 3), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))

classifier.add(Conv2D(256, (3, 3), activation='relu', padding='Same'))
classifier.add(Conv2D(256, (3, 3), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))

classifier.add(Conv2D(512, (3, 3), activation='relu', padding='Same'))
classifier.add(Conv2D(512, (3, 3), activation='relu', padding='Same'))
classifier.add(Conv2D(512, (3, 3), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))

classifier.add(Flatten())

classifier.add(Dense(1024, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Dense(5005, activation='softmax'))

classifier.summary()

reduce = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5,
                           verbose=1, mode='min', min_delta=0.0001, cooldown=1, min_lr=0.0001)

checkpoint = ModelCheckpoint('model.h5', verbose=1, monitor='loss', save_best_only=True, mode='min')

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = classifier.fit(images, Y, epochs=1024, batch_size=64, callbacks=[checkpoint])

plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

del history
del classifier
K.clear_session()
gc.collect()
cuda.select_device(0)
cuda.close()
