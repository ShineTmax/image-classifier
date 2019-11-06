import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
from PIL import Image
import os

os.chdir("../data")

onlyfiles = []
folders = ["train/tulips", "train/sunflower"]
for folder in folders:
    onlyfiles += [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


###pour afficher les images 
from IPython.display import display
from IPython.display import Image as _Imgdis
for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=onlyfiles[i], width=240, height=320))


from scipy import ndimage
from keras.preprocessing.image import  img_to_array, load_img

train_files = []
y_train = []
i=0
for _file in onlyfiles:
    train_files.append(_file)
    if (_file.find("tulips") != -1):
        lable = 0
    else:
        lable = 1
    
    y_train.append(lable)

channels = 3
nb_classes = 1
image_height = 40
image_width = 40
dataset = np.ndarray(shape=(len(train_files), image_width, image_height, channels ),
                     dtype=np.float32)

i = 0
for _file in train_files:
    img = load_img(_file)  # this is a PIL image
    #img = img.thumbnail((image_width, image_height))
    img = img.resize((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    #x = x.reshape((3, 40, 40))
    # Normalize
    #x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
print("dataset chargé")



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset, y_train, test_size=0.2, random_state=33)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=33)
print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, 40, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])


# Train the model
model.fit(X_train / 255.0, tf.keras.utils.to_categorical(Y_train),
          shuffle=True,
          validation_data=(X_test / 255.0, tf.keras.utils.to_categorical(Y_test))
          )


scores = model.evaluate(X_test / 255.0, tf.keras.utils.to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])