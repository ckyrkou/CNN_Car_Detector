from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.initializers import RandomNormal
from keras.regularizers import l2

def make_model():

############################
#### EDIT ONLY THIS BLOCK

    model = Sequential()

    model.add(Conv2D(32, (5, 5),kernel_initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=l2(0.001), input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5),kernel_initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5),kernel_initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, kernel_initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

############################

    # check if model is implemented
    assert type(model) is Sequential, "model not defined"

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model
