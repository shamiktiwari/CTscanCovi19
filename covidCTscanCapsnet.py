# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:52:08 2020

@author: shamik.tiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import DenseNet121
disease_types=['COVID', 'non-COVID']
data_dir = 'E:\covidCTscan'
train_dir = os.path.join(data_dir)
train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}\{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) 
train.head()


IMAGE_SIZE = 64
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) 

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

data = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        data[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
data = data / 255.
print('Train Shape: {}'.format(data.shape))


Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=2)

BATCH_SIZE = 64

X_train, X_1, y_train, y_1 = train_test_split(data, Y_train, test_size=0.3, random_state=27)

X_cv, X_test, y_cv, y_test = train_test_split(X_1, y_1, test_size=0.3, random_state=33)

EPOCHS = 50
SIZE=64
N_ch=3
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
n_class=2
def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)

   conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)

   primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=3, kernel_size=9, strides=2, padding='valid')
   CTscnaCaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,name='CTscnaCaps')(primarycaps)

   out_caps = Length(name='capsnet')(CTscnaCaps)

   y = layers.Input(shape=(n_class,))
   masked_by_y = Mask()([CTscnaCaps, y]) 
   masked = Mask()(CTscnaCaps) 

   decoder = models.Sequential(name='decoder')
   decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
   decoder.add(layers.Dense(1024, activation='relu'))
   decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
   decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

   train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
   eval_model = models.Model(x, [out_caps, decoder(masked)])

   noise = layers.Input(shape=(n_class, 16))
   noised_CTscnaCaps = layers.Add()([CTscnaCaps, noise])
   masked_noised_y = Mask()([noised_CTscnaCaps, y])
   manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
   return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
   
   L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
      0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

   return K.mean(K.sum(L, 1))
Covidmodel, eval_model, manipulate_model = CapsNet(input_shape=X_train.shape[1:],n_class=len(np.unique(np.argmax(y_train, 1))),routings=3)

Covidmodel.compile(optimizer=optimizers.Adam(lr=0.0012), loss=[margin_loss, 'mse'], loss_weights=[1., 0.192],metrics={'capsnet': 'accuracy'})
Covidmodel.summary()
epochs=25
def train_generator(x, y, batch_size, shift_fraction=0.1):
 train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,height_shift_range=shift_fraction) 
 generator = train_datagen.flow(x, y, batch_size=batch_size)
 while 1:
  x_batch, y_batch = generator.next()
  yield ([x_batch, y_batch], [y_batch, x_batch])


aug = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=1, width_shift_range=0.01,
    height_shift_range=0.1, shear_range=0.02, 
    zoom_range=0.02,horizontal_flip=True, vertical_flip=True,
    fill_mode="nearest")

hist = Covidmodel.fit_generator(generator=train_generator(X_train, y_train, 10, 0.01),steps_per_epoch=int(y_train.shape[0] / 10),epochs=epochs,validation_data=[[X_cv,y_cv],[y_cv,X_cv]])

accuracy = hist.history['capsnet_acc']
val_accuracy = hist.history['val_capsnet_acc']
loss = hist.history['capsnet_loss']
val_loss = hist.history['val_capsnet_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()





from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=2
lw=2
y_score = Y_pred
target=y_test
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve for Covid-19 detection model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


y_pred, _ = model.predict([X_1, np.zeros((X_1.shape[0],2))],batch_size = 256, verbose = True)
predicted_classes = y_pred
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
tar = np.argmax(np.round(y_1),axis=1)
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(2)]
print(classification_report(tar, predicted_classes, labels=None, target_names=None))
