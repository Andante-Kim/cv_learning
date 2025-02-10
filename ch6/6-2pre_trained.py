import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

model = VGG16()
image = load_img('elephant.jpg', target_size=(224,224))

image = img_to_array(image)
image= image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

pred = model.predict(image)

# convert the probabilities to class labels
labels = decode_predictions(pred)   # top5
print(labels[0])
label = labels[0][0]                # top1
print(label)
print('%s (%.2f%%)' % (label[1], label[2]*100))
