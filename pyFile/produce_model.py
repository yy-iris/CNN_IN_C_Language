'''
Save CNN network and one sample of train data.

Run one iteration of training of convnet on the MNIST dataset.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D,Activation,LSTM
from keras.models import load_model
from PIL import Image
from keras.models import Model
from PIL import Image as pil_image
import pandas as pd
from keras import backend as K


def my_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def train_model(X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, nb_classes):

    # conv
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = 1, padding='valid'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    model.save('my_model.h5')

    return model


def predict_model(model, x,y):
    y_ = model.predict(x)

    print('real: ' + str(y))
    print('predict: ' + str(y_))


def forward_check(model, layer_name, input):
    # 生成函数模型
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # 导出中间结果
    conv1d_1 = intermediate_layer_model.predict(input)

    return conv1d_1

# ######## ---------------------- for checking ---------------------------------------

model_name = "my_model.h5"
model = load_model(model_name, custom_objects={'my_mape': my_mape})
#print(model.summary())
layer_name = 'dense_2'   # conv2d_1  max_pooling2d_1  flatten_1   dense_1  dense_2

### if the input is csv file
path = "input.csv"
input_height = 24
input_width = 129
x = np.array(pd.read_csv(path))
x = x.reshape(input_height, input_width, 1)
input = x.transpose(2, 0, 1).reshape(1,1,input_height,input_width)
result = forward_check(model, layer_name, input)


### if the input is jpg file
# img = pil_image.open('pic.jpg')
# x = np.asarray(img)
# input = x.reshape((1, 1, x.shape[0], x.shape[1]))
# result = forward_check(model_name, layer_name, input/255)


# ######## --------------------------------- for training and predicting ---------------------------------------
#
# batch_size = 1000
# nb_classes = 10
# nb_epoch = 1
#
# # input image dimensions
# img_rows, img_cols = 28, 28
# # number of convolutional filters to use
# nb_filters = 4
# # size of pooling area for max pooling
# nb_pool = 2
# # convolution kernel size
# nb_conv = 3
#
# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# #
# # ###### ------------------- train model -----------------------------
# # ## conv
# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# ## fc
# # X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
# # X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
#
# model = train_model(X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, nb_classes)


###### ------------------- prediction ---------------------------------

# model = load_model("my_model.h5")
# xinput = X_test[0].reshape(1, 1, img_rows, img_cols).astype('float32')
# xinput /= 255
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# predict_model(model, xinput, Y_test[0])