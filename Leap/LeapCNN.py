import tensorflow
import keras
from keras.models import load_model
from keras import Input, Model
import keras.layers
from kerassurgeon.operations import delete_layer, insert_layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import glob
from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions, preprocess_input
from PIL import Image
import numpy as np
from skimage import transform
import io



model = VGG16(weights="imagenet", include_top=True,
	input_tensor=Input(shape=(129,129,1)))
model.summary()
img = io.imread('fingers/fingers/test/0a4d7cbc-2522-4e51-968a-1a86d3b7ee19_5L.png')
img = transform.resize(img,(129,129))
np_img = np.asarray(img)
np_img = np.expand_dims(np_img,axis=0)
np_img = np.expand_dims(np_img,axis=3)

print(np_img.shape)
classes = model.predict(np_img)
print(decode_predictions(classes,top=3)[0])




