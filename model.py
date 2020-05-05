# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:58:17 2020

@author: Gilian Brouwer
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


"""#create train and val set

import os
import numpy as np
import shutil



path = os.getcwd()
files = [file for file in os.listdir(path) if os.path.isfile(file)]
random_files=np.random.choice(files, int(len(files)*.2), replace=False)


true = os.getcwd()+'\\tulip\\'
target=os.getcwd()+'\\val_set\\tulip\\'

for f in random_files:
    shutil.move(true+f, target)
    
"""

#starting CNN

model=Sequential()

# add first convu layer
model.add(Conv2D(64, (3, 3),input_shape=(128, 128, 3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# add second convulayer

model.add(Conv2D(64, (3, 3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten

model.add(Flatten())

#add layer

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=5, activation='softmax'))

#compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the image

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory('train_set', target_size=(128,128), batch_size=32, class_mode='categorical')
val_set=val_datagen.flow_from_directory('val_set', target_size=(128,128), batch_size=32, class_mode='categorical')

model.fit_generator(train_set,
                         steps_per_epoch = 6651,
                         epochs = 20,
                         validation_data = val_set,
                         validation_steps = 855)

model.save('flower.h5')

import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
imagelist=os.listdir('prediction')
folder_path = 'prediction'

images = []
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
classes = model.predict_classes(images, batch_size=10)
classes=classes.tolist()
label_map = (train_set.class_indices)

data_tuples = list(zip(imagelist, classes))
df= pd.DataFrame(data_tuples, columns=['Name','class'])
df2= pd.DataFrame(label_map, index=[0])
df2.to_csv('dict.csv')
