import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

## set path for images
TRAIN_PATH = './data/train_img/'
TEST_PATH = './data/test_img/'

batch_size = 32 # tune it
epochs = 1 # increase it

## load data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# function to read image
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64,64))
    return img

# load data
train_img, test_img = [],[]
for img_path in tqdm(train['image_id'].values):
    train_img.append(read_img(TRAIN_PATH + img_path + '.png'))

for img_path in tqdm(test['image_id'].values):
    test_img.append(read_img(TEST_PATH + img_path + '.png'))

# normalize images
x_train = np.array(train_img, np.float32) / 255.
x_test = np.array(test_img, np.float32) / 255.

# target variable - encoding numeric value
label_list = train['label'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]
y_train = np.array(y_train)

y_train = to_categorical(y_train)

#Model

## neural net architechture

model = Sequential()
model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = (64,64,3))) # if you resize the image above, shape would be (128,128,3)
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy',  optimizer='adam', metrics = ['accuracy'])

#early_stops = EarlyStopping(patience=3, monitor='val_acc')

#model.fit(x_train, y_train, batch_size=30, epochs=30, validation_split=0.3, callbacks=[early_stops])

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train)

history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=100),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=50)

# Saving Model
model.save('./Models/acc_55.h5')
model.save_weights('./Models/acc_55_weights.h5')
#In JSON Format
json_string = model.to_json()

#model = load_model('my_model.h5')
#model.load_weights('my_model_weights.h5')

# make prediction
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis= 1)

# get predicted labels
y_maps = dict()
y_maps = {v:k for k, v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]

# make submission
sub1 = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels})
sub1.to_csv('./submissions/submission_eight.csv', index=False)
