import tensorflow as tf
import numpy as np
from utils import get_data, preprocess_img, custom_callback
from sklearn.model_selection import train_test_split

images, labels = get_data('Data')
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)

train_images = np.array(list(map(preprocess_img, train_images)))
val_images = np.array(list(map(preprocess_img, val_images)))

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = val_images.reshape(val_images.shape[0], 28, 28, 1)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1),  activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation="relu"),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

cb = custom_callback()
model.fit(datagen.flow(train_images, train_labels),
                                epochs = 30,
                                validation_data = (val_images, val_labels),
                                callbacks = [cb])
model.save("Digit_Recognizer.h5")