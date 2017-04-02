# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:44:16 2017
@author: j13-taniguchi
"""

import os, glob
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from sklearn import cross_validation

import shutil


path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place/project_ssd"
os.chdir(path)

from ssd2 import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
NUM_CLASSES = len(place_list)
input_shape = (300, 300, 3)

priors = pickle.load(open('./param/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

xy = []
X = []
Y = []
# フォント画像のデータを読む
npz = "./param/npz/place_name.npz"

xy = np.load(npz)
X = xy["x"]
Y = xy["y"]
Y = Y.reshape((len(Y), 1, Y.shape[1]))
X /= 255
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y)


path_prefix = './image/'

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.summary()
    # モデルをpngでプロット
    plot(model,
         to_file='./param/learning_place_name_v2.png', 
         show_shapes=True,
         show_layer_names=True)
    
    #model.load_weights('weights_SSD300.hdf5', by_name=True)
    
    #freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
    #          'conv2_1', 'conv2_2', 'pool2',
    #          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']
    
    #for L in model.layers:
    #    if L.name in freeze:
    #        L.trainable = False
    
    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    
    callbacks = [keras.callbacks.ModelCheckpoint('./param/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'),
                 #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
                 ]
    
    base_lr = 3e-4
    optim = keras.optimizers.Adam(lr=base_lr)
    # optim = keras.optimizers.RMSprop(lr=base_lr)
    # optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
    #model.compile(optimizer=optim,
    #              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    '''
    nb_epoch = 30
    history = model.fit_generator(gen.generate(True),
                                  gen.train_batches,
                                  nb_epoch,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  nb_val_samples=gen.val_batches,
                                  nb_worker=1)
    '''
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        nb_epoch=50,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))
    # 学習履歴をプロット        
    plot(model, to_file='./param/learning_place_name_v2_2.png')
    # モデルを保存
    model.save_weights('./param/learning_place_name_v2_2.hdf5')
    # チェックポイントとなっていたファイルを削除
    shutil.rmtree('./param/checkpoints')
    
    # 重みパラメータをJSONフォーマットで出力
    model_json = model.to_json()
    with open('./param/learning_place_name_v2_2.json', 'w') as json_file:
        json_file.write(model_json)
    
    
    inputs = []
    images = []
    preds = model.predict(X_test, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)
    
    
    print("finish!!")

