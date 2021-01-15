import tensorflow
import numpy as np
import os,cv2

def get_data(file_folder):
    classes = 10
    images = []
    labels = []

    for i in range(0,classes):
        imgList = os.listdir(f'{file_folder}/{i}')
        for image in imgList:
            img = cv2.imread(f'{file_folder}/{i}/{image}')
            img = cv2.resize(img,(28,28))
            images.append(img)
            labels.append(i)

    return np.array(images), np.array(labels)

def preprocess_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    x,img=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img=img/255
    return img


def display_predList(predList):
    predicted_digits = []
    for i in range(len(predList)):
        for j in range(len(predList)):
            predicted_digits.append(predList[j][i])

    return predicted_digits

class custom_callback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True