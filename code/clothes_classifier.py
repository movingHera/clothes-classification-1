__autuor__ = 'Zhenyuan Shen'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time, gc, imutils, cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras import optimizers

# from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.model_selection import KFold

from helpers import *
import newnet
import math, random
import preprocess

class ClothesTypeClassifier():
    def __init__(self, input_dim=128, batch_size=8, nfolds=1, epochs=20, learn_rate=1e-4):
        self.input_dim = input_dim  # 197
        self.batch_size = batch_size
        self.nfolds = nfolds
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.model_path = '../weights/clothes_classifier_model.h5'
        self.load_data(0)
        self.nTTA = 2
        self.nAug = 2
        # self.model = newnet.denseNet121(self.input_dim, self.numCategory)
        # self.model = newnet.model3(self.input_dim, self.numCategory)
        self.model = newnet.vgg16(self.input_dim, self.numCategory)

        with open('../weights/model.json', 'w') as json_file:
            json_file.write(self.model.to_json())

    def load_data(self, source=1):
        if source == 1:
            self.numCategory, self.label_map, self.inv_label_map, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = \
                preprocess.processing(
                    input_dir='../input/Category and Attribute Prediction Benchmark/Img',
                    labels_dir=r'../list_attr_cloth.txt',
                    gt_dir=r'../list_attr_img.txt'
                )
        else:
            self.numCategory, self.label_map, self.inv_label_map, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = \
                preprocess.processing(
                    input_dir='../input/Category and Attribute Prediction Benchmark/Img'
                )
        # labels = preprocess.load_labels(label_filepath=r'../list_attr_cloth.txt')
        # self.label_map = {l: i for i, l in enumerate(self.labels)}
        # self.inv_label_map = {i: l for l, i in self.label_map.items()}
        # print(self.inv_label_map)

    def train(self):
        nTrain = len(self.x_train)
        nVal = len(self.x_val)
        print('Training on {} samples'.format(nTrain))
        print('Validating on {} samples'.format(nVal))

        train_index = list(range(nTrain))
        val_index = list(range(nVal))
        random.shuffle(train_index)
        random.shuffle(val_index)

        x_train = self.x_train[train_index]
        y_train = self.y_train[train_index]

        x_val = self.x_val[val_index]
        y_val = self.y_val[val_index]

        thres = []
        def valid_generator():
            while True:
                for start in range(0, nVal, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nVal)
                    img_val_paths = x_val[start:end]
                    img_val_labels = y_val[start:end]
                    for i in range(len(img_val_paths)):
                        img = cv2.imread(img_val_paths[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim))
                        img = transformations2(img, np.random.randint(self.nAug))
                        # targets = np.zeros(self.numCategory)
                        # for j in img_val_labels[i]:
                        #     targets[j] = 1

                        targets = img_val_labels[i]

                        x_batch.append(img)
                        y_batch.append(targets)
                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nTrain)
                    img_train_paths = x_train[start:end]
                    img_train_labels = y_train[start:end]
                    for i in range(len(img_train_paths)):
                        img = cv2.imread(img_train_paths[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim))
                        img = transformations2(img, np.random.randint(self.nAug))
                        # targets = np.zeros(self.numCategory)
                        # for j in img_train_labels[i]:
                        #     targets[j] = 1

                        targets = img_train_labels[i]

                        x_batch.append(img)
                        # print(targets)
                        y_batch.append(targets)
                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch


        opt  = optimizers.RMSprop(lr=self.learn_rate)
        self.model.compile(loss=['binary_crossentropy'], # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])

        # callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        #              EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0),
        #              ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=2,
                                   verbose=1,
                                   min_delta=1e-4),
                    ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=2,
                                           cooldown=2,
                                           verbose=1),
                    ModelCheckpoint(filepath=self.model_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True)]

        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_generator(),
            validation_steps=math.ceil(nVal / float(self.batch_size)))

        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)

        p_valid = self.model.predict_generator(generator=valid_generator(),
                                              steps=math.ceil(nVal / float(self.batch_size)))

        ## find best thresholds for each class
        y_valid = []
        for i in range(nVal):
            # targets = np.zeros(self.numCategory)
            # for j in self.y_val[i]:
            #     targets[j] = 1
            targets = self.y_val[i]

            y_valid.append(targets)
        y_valid = np.array(y_valid, np.uint8)

        best_threshold, best_scores = find_f_measure_threshold2(p_valid, y_valid, num_iters=20)
        thres.append(best_threshold)
        val_score = best_scores[-1]

        return thres, val_score


    def test(self, thres, val_score, early_fusion=True, one_hot=False):
        nTest = len(self.x_test)

        print('Testing on {} samples'.format(nTest))

        def test_generator(transformation):
            while True:
                for start in range(0, nTest, self.batch_size):
                    x_batch = []
                    end = min(start + self.batch_size, nTest)
                    for i in range(start, end):
                        img = cv2.imread(self.x_test[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim))
                        img = transformations2(img, transformation)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32) / 255.0
                    yield x_batch

        y_full_test = []

        if not os.path.isfile(self.model_path):
            return RuntimeError('No model file exists.')
        self.model.load_weights(self.model_path)

        # n-fold TTA
        p_full_test = []
        for i in range(self.nTTA):
            p_test = self.model.predict_generator(generator=test_generator(transformation=i),
                                                     steps=math.ceil(nTest / float(self.batch_size)))
            p_full_test.append(p_test)

        p_test = np.array(p_full_test[0])
        for i in range(1, self.nTTA):
            p_test += np.array(p_full_test[i])
        p_test /= self.nTTA
        y_full_test.append(p_test)

        raw_result = np.zeros(y_full_test[0].shape)
        if early_fusion:
            thresh = np.zeros([1, len(thres[0])])
            for i in range(self.nfolds):
                raw_result += y_full_test[i]
                thresh += thres[i]

            raw_result /= float(self.nfolds)
            thresh /= float(self.nfolds)
            if one_hot:
                id_max = np.argmax(raw_result, axis=1)
                result = np.zeros(raw_result.shape)
                for i,j in enumerate(id_max):
                    result[i, j] = 1
            else:
                result = (raw_result > thresh)
        else:
            for i in range(self.nfolds):
                if one_hot:
                    id_max = np.argmax(y_full_test[i], axis=1)
                    res = np.zeros(y_full_test.shape)
                    for i,j in enumerate(id_max):
                        res[i,j] = 1
                    raw_result += res
                else:
                    raw_result += (y_full_test[i] > thres[i])
            result = raw_result / float(self.nfolds)


        y_test = []
        for i in range(nTest):
            # targets = np.zeros(self.numCategory)
            # for j in self.y_test[i]:
            #     targets[j] = 1
            targets = self.y_test[i]
            y_test.append(targets)
        y_test = np.array(y_test, np.uint8)
        f2 = fbeta_score(y_test, result, beta=2, average='samples')

        print('-------------------------------------')
        print('F2-score: {}'.format(f2))

        # final results
        preds = []
        for index in range(result.shape[0]):
            pred = ' '.join(list(map(lambda x: self.inv_label_map[x], *np.where(result[index, :] == 1))))
            if len(pred) == 0:
                if early_fusion:
                    pred = ' '.join(
                        list(map(lambda x: self.inv_label_map[x], *np.argmax(raw_result[index, :] - thresh))))
                else:
                    pred = ' '.join(list(map(lambda x: self.inv_label_map[x], *np.argmax(raw_result[index, :]))))
            preds.append(pred)

        df_test_data = pd.DataFrame()
        df_test_data['image_names'] = self.x_test
        df_test_data['tags'] = preds
        df_test_data.to_csv('../result_{}.csv'.format(val_score), index=False)

if __name__ == "__main__":
    af = ClothesTypeClassifier()

    thresh, val_score = af.train()
    # thresh, val_score = load_param()
    print("thresh:\n{}".format(thresh))
    print("val_score:", val_score)

    af.test(thres=thresh, val_score=val_score, early_fusion=True)

