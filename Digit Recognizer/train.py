import cv2
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import gc
from keras import backend as K
from numba import cuda
import tensorflow as tf
from keras.utils import to_categorical

K.clear_session()
gc.collect()
cuda.select_device(0)
cuda.close()

K.tensorflow_backend._get_available_gpus()

K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(
    log_device_placement=True, gpu_options=tf.GPUOptions(allow_growth=True), device_count={'GPU': 1, 'CPU': 1})))

plt.style.use('ggplot')

train_data = pd.read_csv("train.csv")

Y = train_data["label"]

X = train_data.drop(labels=["label"], axis=1)

Y = to_categorical(Y, num_classes=10)

X = X / 255.0

X = X.values.reshape(-1, 28, 28, 1)

print(X.shape)
print(" ")
print(Y.shape)

classifier = Sequential()

classifier.add(Conv2D(16, (7, 7), input_shape=(28, 28, 1), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Conv2D(32, (5, 5), activation='relu', padding='Same'))
classifier.add(Conv2D(32, (5, 5), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Dropout(0.1))
classifier.add(BatchNormalization())

classifier.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(10, activation='softmax'))

classifier.summary()

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = classifier.fit(X, Y, epochs=128, batch_size=64)

classifier.save("model.h5")

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
