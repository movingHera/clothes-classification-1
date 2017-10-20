# -*- coding: utf-8 -*-

from keras.models import model_from_json
import cv2
import numpy as np
import keras.backend as K
from glob import glob
import pandas as pd

import time
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5



global model
class ClothesEncoding():
    def __init__(self, model_file, weight_file):
        self.model_file = model_file
        self.weight_file = weight_file
        self.model = None
        self.init_model()

    def init_model(self):
        '''
        load model from file
        :param model_file: json file that defines the network graph
        :param weight_file: weight file that stores the weights of the graph
        :return: model object
        '''

        with open(self.model_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            if not os.path.exists(self.weight_file):
                raise RuntimeError('No weights file founded.')
            self.model.load_weights(self.weight_file)
            self.graph = tf.get_default_graph()


    def inference(self, image):
        '''
        predict the mask given specific image and the model object
        :param image: [512, 512, 3] 'float32' input image normalized to [0, 1]
        :param model: model object
        :return: [1, 512, 512, 1] 'float32' output mask
        '''
        # print image.shape
        # print image.dtype

        m_Height, m_Width = image.shape[:2]
        img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, *img.shape)

        with self.graph.as_default():
            l = self.model.get_layer('clothes_encoding')
            # node_index = model.layers.index(l)   # node_index is 63

            func = K.function([self.model.input, K.learning_phase()], [l.output])
            encode = func([img, 1.])[0]
            return encode


if __name__ == '__main__':
    feat_extractor = ClothesEncoding('../weights/model.json', '../weights/clothes_classifier_model.h5')
    img = cv2.imread('/home/jin/Desktop/test.jpg')
    encode = feat_extractor.inference(img)
    print('encode of the clothes:\n', encode)


    # generate features
    def getList(path, suffix_tuple=('jpg', 'png')):
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(suffix_tuple):
                    if file not in file_list:
                        f = os.path.join(root, file)
                        file_list.append(f)
        return file_list

    data_files = glob(os.path.join("../input/crop_image/img/25_Mesh-Paneled_Jersey_Dress", "*.jpg"))
    # data_files = getList("../input/", "jpg")

    images = []
    features = []
    i = 1
    N = len(data_files)
    for image_path in data_files:
        i += 1
        print("{}/{}".format(i, N))
        images.append(image_path[image_path.rfind('/')+1:])

        img = cv2.imread(image_path)
        encode = feat_extractor.inference(img)  # encode is a numpy array
        features.append(','.join(map(str, encode.flatten().tolist())))
        if i > 100:
            break;

    print("Generating submission file...")
    df = pd.DataFrame({'clothes': images, 'features': features})
    df.to_csv('../submit/clothes-features.csv', index=False)