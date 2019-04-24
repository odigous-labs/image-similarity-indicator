
'''
Within this script, we design hidden layers of our siamese model which depend on two feature vectors of a pre-defined model (We used VGG16 model at this moment).
----
If need to use this script within another code then can import the script and call the functions with relevant arguments.

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0

For the contrastive loss can refer the following resources:
    From Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
'''

from keras import backend as K
from keras.layers import BatchNormalization, Conv2D, Flatten
from keras.layers.core import Activation, Dense, Dropout, Lambda
import os

def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)


def cosine_distance_output_shape(shapes):
    return shapes[0]


def siamese_model(vector_1, vector_2):

    print('vector 1:', vector_1)
    print('vector 2: ', vector_2)

    merged = Lambda(cosine_distance,
                    output_shape=cosine_distance_output_shape)([vector_1, vector_2])
    print('merge', merged)
    fc1 = Dense(512, kernel_initializer="glorot_uniform",  activation = 'relu')(merged)
    fc1 = Dropout(0.3)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation("relu")(fc1)

    # conv = Conv2D(64, kernel_size=3, activation='relu')(fc1)
    # # conv = Conv2D(32, kernel_size=3, activation='relu')(conv)
    # fc1 = Flatten()(conv)
    fc2 = Dense(256, kernel_initializer="glorot_uniform",  activation = 'relu')(fc1)
    fc2 = Dropout(0.2)(fc2)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation("relu")(fc2)


    fc3 = Dense(128, kernel_initializer="glorot_uniform",  activation = 'relu')(fc2)
    fc3 = Dropout(0.2)(fc3)
    fc3 = BatchNormalization()(fc3)
    fc3 = Activation("relu")(fc3)

    # pred = Dense(1, kernel_initializer="glorot_uniform", activation = 'softmax')(fc2)
    pred = Dense(1, kernel_initializer="normal", activation='sigmoid')(fc3)
    #pred = Activation("softmax")(pred)
    print('pred', pred)
    return pred


def contrastive_loss(y_true, y_pred):

    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


