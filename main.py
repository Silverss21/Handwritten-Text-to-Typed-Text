import numpy as np
import pandas as pd
import os
import random
import time
import cv2
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras import optimizers, utils
from keras.preprocessing import image
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

# Characters dataset
IMG_SIZE = 32
CHAR_TRAIN_PATH = "./dataset/characters/Train/"
CHAR_VAL_PATH = "./dataset/characters/Validation/"
# Excldue these characters in model training
EXCLUDE_DIR = ["#","$","&","@","0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Single character max samples
MAX_SINGLE_CHAR_SAMPLES_TRAIN = 4000
MAX_SINGLE_CHAR_SAMPLES_VAL = 1000

train_data = []
for i in os.listdir(CHAR_TRAIN_PATH):
    if i in EXCLUDE_DIR:
        continue
        
    count = 0
    sub_directory = os.path.join(CHAR_TRAIN_PATH,i)
    for j in os.listdir(sub_directory):
        count+=1
        if count > MAX_SINGLE_CHAR_SAMPLES_TRAIN:
            break
        img = cv2.imread(os.path.join(sub_directory,j),0)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        train_data.append([img,i])

len(train_data)

val_data = []
for i in os.listdir(CHAR_VAL_PATH):
    if i in EXCLUDE_DIR:
        continue
    count = 0
    sub_directory = os.path.join(CHAR_VAL_PATH,i)
    for j in os.listdir(sub_directory):
        count+=1
        if count > MAX_SINGLE_CHAR_SAMPLES_VAL:
            break
        img = cv2.imread(os.path.join(sub_directory,j),0)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        val_data.append([img,i])

len(val_data)

random.shuffle(train_data)
random.shuffle(val_data)

train_X = []
train_Y = []
for features,label in train_data:
    train_X.append(features)
    train_Y.append(label)

val_X = []
val_Y = []
for features,label in val_data:
    val_X.append(features)
    val_Y.append(label)

LB = LabelBinarizer()
train_Y = LB.fit_transform(train_Y)
val_Y = LB.fit_transform(val_Y)

train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,32,32,1)
train_Y = np.array(train_Y)

val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1,32,32,1)
val_Y = np.array(val_Y)

print(train_X.shape,val_X.shape)

print(train_Y.shape,val_Y.shape)

model_1 = Sequential()

# Convolutional part

model_1.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(32,32,1)))
model_1.add(MaxPooling2D(pool_size=(2,2)))

model_1.add(Conv2D(64, (3, 3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))

model_1.add(Conv2D(128, (3, 3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))

# Classification part

model_1.add(Dropout(0.25))
 
model_1.add(Flatten())
model_1.add(Dense(128, activation='relu'))
model_1.add(Dropout(0.2))

model_1.add(Dense(26, activation='softmax'))

model_1.summary()

# Compile model using "adam" optimizer
model_1.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

model_2 = Sequential()

# Convolutional part

model_2.add(Conv2D(32, (3, 3), padding = "same", activation='leaky_relu', input_shape=(32,32,1)))
model_2.add(MaxPooling2D(pool_size=(2,2)))

model_2.add(Conv2D(64, (3, 3), activation='leaky_relu'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

model_2.add(Conv2D(96, (3, 3), activation='leaky_relu'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

# Classification part

model_2.add(Dropout(0.25))
 
model_2.add(Flatten())
model_2.add(Dense(64, activation='elu'))
model_2.add(Dropout(0.2))

model_2.add(Dense(26, activation='softmax'))

model_2.summary()

model_2.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

model_3 = Sequential()

# Convolutional part

model_3.add(Conv2D(32, (5, 5), padding = "same", activation='leaky_relu', input_shape=(32,32,1)))
model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Conv2D(64, (5, 5), activation='leaky_relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Conv2D(128, (5, 5), activation='leaky_relu'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

# Classification part

model_3.add(Dropout(0.25))
 
model_3.add(Flatten())
model_3.add(Dense(128, activation='elu'))
model_3.add(Dropout(0.2))

model_3.add(Dense(26, activation='softmax'))

model_3.summary()

model_3.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

train_X.shape

train_Y.shape

val_X.shape

val_Y.shape

start1 = time.time()
history_1 = model_1.fit(train_X,train_Y, epochs=50, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)
end1 = time.time()
length1 = end1 - start1
print("Model 1 took", round(length1 / 60), "mins ")

start2 = time.time()
history_2 = model_2.fit(train_X,train_Y, epochs=50, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)
end2 = time.time()
length2 = end2 - start2
print("Model 2 took", round(length2 / 60), "mins ")

start3 = time.time()
history_3 = model_3.fit(train_X,train_Y, epochs=50, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)
end3 = time.time()
length3 = end3 - start3
print("Model 3 took", round(length3 / 60), "mins ")

plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('Model 1 - Training vs Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('Model 2 - Training vs Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('Model 3 - Training vs Validation accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('Model 1 - Training vs Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('Model 2 - Training vs Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('Model 3 - Training vs Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

ypred = model_1.predict(val_X, batch_size=1, verbose=1)

ypred2 = model_2.predict(val_X, batch_size=1, verbose=1)

ypred3 = model_3.predict(val_X, batch_size=1, verbose=1)

def showConfuctionMatrix(yp, yt, title = "Confusion matrix"):
    y_pred = np.argmax(yp, axis=1)
    y_test = np.argmax(yt, axis=1)
    vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    voc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    annot = np.tile(voc, (26, 1))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(cm, ax=ax, xticklabels=voc, yticklabels=voc)
    plt.title(title, fontsize = 20) # title with fontsize 20
    plt.xlabel('Actual', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Predicted', fontsize = 15) # y-axis label with fontsize 15
    
    plt.show()

showConfuctionMatrix(ypred, val_Y, "Model 1 - Confusion Matrix")

showConfuctionMatrix(ypred2, val_Y, "Model 2 - Confusion Matrix")

showConfuctionMatrix(ypred3, val_Y, "Model 3 - Confusion Matrix")

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_letters(img, model_nr = 1):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            # Mark the contour by placing green rectangle as border
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)
        if (model_nr == 1):
            ypred = model_1.predict(thresh)
        else:
            ypred = model_2.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)
    return letters, image

#plt.imshow(image)

def get_word(letter):
    word = "".join(letter)
    return word

letter,image = get_letters("./dataset/words/train_v2/train/TRAIN_00003.jpg")
word = get_word(letter)
print(word)
plt.imshow(image)

letter,image = get_letters("./dataset/words/train_v2/train/TRAIN_00023.jpg")
word = get_word(letter)
print(word)
plt.imshow(image)

letter,image = get_letters("./dataset/words/train_v2/train/TRAIN_00030.jpg")
word = get_word(letter)
print(word)
plt.imshow(image)

letter,image = get_letters("./dataset/words/validation_v2/validation/VALIDATION_0005.jpg")
word = get_word(letter)
print(word)
plt.imshow(image)

letter,image = get_letters("./dataset/words/test_v2/test/TEST_0248.jpg")
word = get_word(letter)
print(word)
plt.imshow(image)
