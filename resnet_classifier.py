# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:34:23 2019

@author: ahall
"""

import tensorflow
import keras



IMG_SIZE=71
model = keras.applications.xception.Xception(weights = None, include_top=True, input_shape= (IMG_SIZE,IMG_SIZE,3),classes=2)

def build_data_generator(phase):
    datagen = keras.preprocessing.image.ImageDataGenerator()
    
    data_generator = datagen.flow_from_directory(
    #directory="C:\\Users\\ahall\\Documents\\\projects\\fossil hackathon\\labelled_dataset\\"+ phase ,
    directory="C:\\Users\\ahall\\Documents\\\projects\\fossil hackathon\\trial_dataset\\"+ phase ,
    #directory = "C:\\Users\\ahall\\Documents\\projects\\generic_CNN\\catdogdata\\" + phase,
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validate_generator.n//validate_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)


