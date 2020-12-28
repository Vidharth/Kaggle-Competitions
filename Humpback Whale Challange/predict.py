from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
from keras.utils import to_categorical
import pandas as pd
from numba import cuda
from keras import backend as K
import gc


cuda.select_device(0)
cuda.close()
K.clear_session()
gc.collect()

file = os.listdir("D:/STUDY/PROJECTS/humpback-whale-identification/test")
print(len(file))
images = np.zeros((len(file), 64, 64, 3))
print(images.shape)
x = 0
for img in (file):
    pic = image.load_img("D:/STUDY/PROJECTS/humpback-whale-identification/test/"+img, target_size=(64, 64, 3))
    n = image.img_to_array(pic)
    n = preprocess_input(n)
    images[x] = n
    if (x % 500 == 0):
            print("Processing image: ", x+1, ", ", img)
    x += 1

images /= 255.0

model = load_model("model.h5")

predictions = model.predict(np.array(images), verbose=1)

data = pd.read_csv('train.csv')
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data["Id"])

test_data = pd.DataFrame(file, columns=['Image'])
test_data['Id'] = ''

for i, pred in enumerate(predictions):
    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

print(test_data.head(10))
print(test_data.shape)
test_data.to_csv('submissions.csv', index=False)


K.clear_session()
gc.collect()
cuda.select_device(0)
cuda.close()
