# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:23:34 2019

@author: ahall
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:34:23 2019

@author: ahall
"""

import tensorflow
import keras



IMG_SIZE=64
model = keras.Sequential()

model.add(keras.layers.Conv2D(128, kernel_size=3,activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(256, kernel_size=3, padding="same", activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(512, kernel_size=3, padding="same", activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=100, activation='relu'  ))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=50, activation='relu'  ))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=25, activation='relu'  ))
model.add(keras.layers.Dropout(0.3))

from keras.layers.core import Activation

model.add(keras.layers.Dense(2))
model.add(Activation("softmax"))

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



def build_data_generator(phase):
    datagen = keras.preprocessing.image.ImageDataGenerator()
    
    data_generator = datagen.flow_from_directory(
    #directory="C:\\Users\\ahall\\Documents\\\projects\\fossil hackathon\\labelled_dataset\\"+ phase ,
    #directory="C:\\Users\\ahall\\Documents\\\projects\\fossil hackathon\\trial_dataset\\"+ phase ,
    directory = "C:\\Users\\ahall\\Documents\\projects\\generic_CNN\\catdogdata\\" + phase,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42)
    
    return data_generator


train_generator = build_data_generator("train")
validate_generator = build_data_generator("validate")
#test_generator = build_data_generator("test")

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validate_generator.n//validate_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)


