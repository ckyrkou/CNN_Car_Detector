import cv2
import numpy as np
import random as rnd

from keras.callbacks import ModelCheckpoint
from make_model import *

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

seed = 11
rnd.seed(seed)
np.random.seed(seed)


############################
#### EDIT ONLY THIS BLOCK

model = make_model()
epochs = 100
winH,winW = 50,50

############################

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
		rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(winH, winW),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(winH, winW),
        batch_size=batch_size,
        class_mode='binary')

filepath="weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

class_weight = {0: 10,
                1: 1}

# Change steps_per_epoch and validation_steps according to the dataset that you use.
				
model.fit_generator(
        train_generator,
        steps_per_epoch=5131 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1603 // batch_size,
        callbacks=callbacks_list,
        class_weight=class_weight)
