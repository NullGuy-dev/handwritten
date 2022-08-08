import os
from tkinter import image_names
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction import image
import tensorflow as tf

model = tf.keras.models.load_model('handwritten.model')

image_number = 1
# digits is a folder with images of numbers, add images to this folder
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        # name the images like 1.png, 2.png and so on
        img = cv2.imread(f"digits/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("ERROR")
    finally:
        image_number += 1
