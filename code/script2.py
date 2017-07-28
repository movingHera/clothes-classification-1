# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../data"]).decode("utf8"))

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
import math

class AmazonForest():
    def __init__(self, input_dim=256, batch_size=16, nfolds=5, epochs=300, learn_rate=1e-4):
        self.input_dim = input_dim  # 197
        self.batch_size = batch_size
        self.nfolds = nfolds
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.model = newnet.denseNet121(input_dim)
        self.load_data()

    def load_data(self):
        self.df_train_data = pd.read_csv('../data/train_v2.csv') # [:100]
        self.df_test_data = pd.read_csv('../data/sample_submission_v2.csv') # [:100]

        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in self.df_train_data['tags'].values])))

        self.label_map = {l: i for i, l in enumerate(labels)}
        self.inv_label_map = {i: l for l, i in self.label_map.items()}
        # self.inv_label_map = {
        #     0: 'slash_burn', 1: 'clear', 2: 'blooming', 3: 'primary', 4: 'cloudy', 5: 'conventional_mine', 6: 'water',
        #     7: 'haze', 8: 'cultivation', 9: 'partly_cloudy', 10: 'artisinal_mine', 11: 'habitation', 12: 'bare_ground',
        #     13: 'blow_down', 14: 'agriculture', 15: 'road', 16: 'selective_logging'
        # }

    def initialize(self, augmentation = True):
        print ("inv_label_map:\n{}".format(inv_label_map))
        x_train = []
        y_train = []
        x_val = []
        y_val = []

        for f, tags in tqdm(self.df_train.values, miniters=1000):
            img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
            targets = np.zeros(17)
            Tags = tags.split(' ')
            aug_flip = False
            aug_flip2 = False
            for t in Tags:
                targets[self.label_map[t]] = 1
                if not aug_flip and t in ['conventional_mine', 'blow_down', 'slash_burn', 'blooming', 'artisinal_mine',
                                          'selective_logging', 'bare_ground']:
                    aug_flip = True
                if not aug_flip2 and t in ['cloudy', 'haze', 'habitation', 'cultivation']:
                    aug_flip2 = True

            img = cv2.resize(img, (self.input_dim, self.input_dim))


            # if(len(x_train) > 1000):
            #     break

            # if aug_flip:
            #     nAug = 8
            # elif aug_flip2:
            #     nAug = 4
            # else:
            #     nAug = 1
            #
            # for _ in xrange(nAug):
            #     x_train.append(img)
            #     y_train.append(targets)

            # if aug_flip:
            #     for angle in np.arange(0, 360, 90):
            #         new_img = imutils.rotate(img, angle)
            #         x_train.append(new_img)
            #         y_train.append(targets)
            #
            #         x_train.append(cv2.flip(new_img, 1))
            #         y_train.append(targets)
            # elif aug_flip2:
            #     for angle in np.arange(0, 360, 90):
            #         new_img = imutils.rotate(img, angle)
            #         x_train.append(new_img)
            #         y_train.append(targets)
            # else:
            #     x_train.append(img)
            #     y_train.append(targets)

            x_train.append(img)
            y_train.append(targets)
            if aug_flip:
                x_train.append(cv2.flip(img, 1))
                y_train.append(targets)

            # if np.random.rand() > 0.8:
            #     x_val.append(img)
            #     y_val.append(targets)
            # else:
            #     x_train.append(img)
            #     y_train.append(targets)
                # for angle in np.arange(0, 360, 90):
                #     new_img = imutils.rotate(img, angle)
                #     x_train.append(new_img)
                #     y_train.append(targets)
                #
                #     # x_train.append(cv2.flip(new_img, 1))
                #     # y_train.append(targets)

            # # loop over the rotation angles
            # for angle in np.arange(0, 1+360*augmentation, 90):
            #     new_img = imutils.rotate(img, angle)
            #     x_train.append(new_img)
            #     y_train.append(targets)
            #     if aug_flip:
            #         x_train.append(cv2.flip(new_img, 1))
            #         y_train.append(targets)
            #
            #     # elif aug_flip2:
            #     #     x_train.append(new_img)
            #     #     y_train.append(targets)

        gc.collect()

        # return np.array(x_train, np.float16) / 255.0, \
        #        np.array(y_train, np.uint8), \
        #        np.array(x_val, np.float16) / 255.0, \
        #        np.array(y_val, np.uint8)


        x_train = np.array(x_train, np.uint8)
        y_train = np.array(y_train, np.uint8)
        # x_val = np.array(x_val, np.uint8)
        # y_val = np.array(y_val, np.uint8)
        # return x_train, y_train, x_val, y_val
        return x_train, y_train


    def train(self):
        x_train, y_train = self.initialize()

        # mean = np.round(comp_mean(x_train)).astype('uint8')
        # for i in xrange(len(x_train)):
        #     x_train[i] -= mean

        print(x_train.shape)
        print(y_train.shape)

        num_fold = 0
        kf = KFold(len(y_train), n_folds=self.nfolds, shuffle=True, random_state=1)
        # kf = StratifiedKFold(y_train, n_folds=nfolds, shuffle=True, random_state=1)

        thres = []
        val_score = 0
        for train_index, test_index in kf:
            start_time_model_fitting = time.time()

            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_valid = x_train[test_index]
            Y_valid = y_train[test_index]

            num_fold += 1
            print('Start KFold number {} from {}'.format(num_fold, self.nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))

            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

            # epochs_arr = [50, 50, 50]
            # learn_rates = [0.001, 0.0001, 0.0001]

            epochs_arr = [100]
            learn_rates = [0.0001]

            for learn_rate, epochs in zip(learn_rates, epochs_arr):
                opt = optimizers.Adam(lr=learn_rate)
                self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                              optimizer=opt,
                              metrics=['accuracy'])
                # callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

                callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
                             EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=0),
                             ModelCheckpoint(kfold_weights_path, monitor='val_acc', save_best_only=True, verbose=0)]

                self.model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                      batch_size=self.batch_size,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)

            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)

            p_valid = self.model.predict(X_valid, batch_size = self.batch_size, verbose=2)

            ## find best thresholds for each class
            best_threshold, best_scores = find_f_measure_threshold2(p_valid, Y_valid)

            thres.append(best_threshold)
            val_score += best_scores[-1]

        return thres, val_score/float(self.nfolds)


    def train_new(self):

        # train_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     zoom_range=0.15,
        #     rotation_range=360,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1
        # )
        # val_datagen = ImageDataGenerator(rescale=1. / 255)

        num_fold = 0

        thres = []
        val_score = 0
        # kf = KFold(len(self.df_train_data), n_folds=self.nfolds, shuffle=True, random_state=1)  # deprecated KFold
        kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=1)

        for train_index, test_index in kf.split(self.df_train_data):
            num_fold += 1

            # train_datagen.fit(x_train, augment=True, rounds=2, seed=1)
            # train_generator = train_datagen.flow(x_train[train_index], y_train[train_index], shuffle=True, batch_size=batch_size, seed=int(time.time()))
            # val_generator = val_datagen.flow(x_train[test_index], y_train[test_index], shuffle=False, batch_size=batch_size)

            df_valid = self.df_train_data.ix[test_index]
            print('Validating on {} samples'.format(len(df_valid)))

            def valid_generator():
                while True:
                    for start in range(0, len(df_valid), self.batch_size):
                        x_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, len(df_valid))
                        df_valid_batch = df_valid[start:end]
                        for f, tags in df_valid_batch.values:
                            img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
                            img = cv2.resize(img, (self.input_dim, self.input_dim))
                            img = transformations(img, np.random.randint(6))
                            targets = np.zeros(17)
                            for t in tags.split(' '):
                                targets[self.label_map[t]] = 1
                            x_batch.append(img)
                            y_batch.append(targets)
                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.uint8)
                        yield x_batch, y_batch

            df_train = self.df_train_data.ix[train_index]
            print('Training on {} samples'.format(len(df_train)))

            def train_generator():
                while True:
                    for start in range(0, len(df_train), self.batch_size):
                        x_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, len(df_train))
                        df_train_batch = df_train[start:end]
                        for f, tags in df_train_batch.values:
                            img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
                            img = cv2.resize(img, (self.input_dim, self.input_dim))
                            img = transformations(img, np.random.randint(6))
                            targets = np.zeros(17)
                            for t in tags.split(' '):
                                targets[self.label_map[t]] = 1
                            x_batch.append(img)
                            y_batch.append(targets)
                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.uint8)
                        yield x_batch, y_batch

            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

            opt  = optimizers.Adam(lr=self.learn_rate)
            self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])

            # callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
            #              EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0),
            #              ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

            callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=4,
                                       verbose=1,
                                       min_delta=1e-4),
                         ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=2,
                                           cooldown=2,
                                           verbose=1),
                         ModelCheckpoint(filepath=kfold_weights_path,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True)]

            self.model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=math.ceil(len(df_train) / float(self.batch_size)),
                epochs=self.epochs,
                verbose=2,
                callbacks=callbacks,
                validation_data=valid_generator(),
                validation_steps=math.ceil(len(df_valid) / float(self.batch_size)))

            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)

            # p_valid = model.predict_generator(
            #     val_generator, steps=len(test_index)/batch_size,
            # )
            # p_valid = model.predict(val_generator.x/255.0, batch_size = batch_size, verbose=2)
            p_valid = self.model.predict_generator(generator=valid_generator(),
                                              steps=math.ceil(len(df_valid) / float(self.batch_size)))

            ## find best thresholds for each class
            y_valid = []
            for f, tags in df_valid.values:
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[self.label_map[t]] = 1
                y_valid.append(targets)
            y_valid = np.array(y_valid, np.uint8)

            best_threshold, best_scores = find_f_measure_threshold2(p_valid, y_valid)
            thres.append(best_threshold)
            val_score += best_scores[-1]

        return thres, val_score/float(self.nfolds)

    def refine(self, thresh, val_score):
        x_train, y_train = self.initialize()

        # print(x_train.shape)
        # print(y_train.shape)

        kfold_weights_path = os.path.join('', 'weights_kfold_all.h5')

        # epochs = 30
        # learn_rate = 0.0001
        #
        # opt  = optimizers.Adam(lr=learn_rate)
        # model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
        #                 optimizer=opt,
        #                 metrics=['accuracy'])
        #
        # callbacks = [ModelCheckpoint(kfold_weights_path, monitor='val_acc', save_best_only=True, verbose=0)]
        #
        # model.fit(x = x_train, y= y_train,
        #               batch_size=batch_size,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)

        if os.path.isfile(kfold_weights_path):
            self.model.load_weights(kfold_weights_path)

        x_test = []
        df_test = pd.read_csv('../data/sample_submission_v2.csv')

        augmentation = 4
        mirror = False

        for f, tags in tqdm(df_test.values, miniters=1000):
            if f.startswith('test'):
                img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
                img = cv2.resize(img, (self.input_dim, self.input_dim))
                for angle in np.arange(0, augmentation*90, 90):
                    new_img = imutils.rotate(img, angle)
                    x_test.append(new_img)
                    if mirror:
                        x_test.append(cv2.flip(new_img, 1))
            else:
                img = cv2.imread('../data/test-jpg-additional/{}.jpg'.format(f))
                img = cv2.resize(img, (self.input_dim, self.input_dim))
                for angle in np.arange(0, augmentation*90, 90):
                    new_img = imutils.rotate(img, angle)
                    x_test.append(new_img)
                    if mirror:
                        x_test.append(cv2.flip(new_img, 1))

        x_test  = np.array(x_test, np.float16)/255.0
        # for i in xrange(len(x_test)):
        #     x_test[i] -= mean

        ntimes = augmentation * (1 + mirror)
        nsamples = len(df_test)

        p_test = self.model.predict(x_test, batch_size = self.batch_size, verbose=2)
        if ntimes > 1:
            for j in xrange(nsamples):
                p_test[j,:] = np.mean(p_test[j*ntimes:(j+1)*ntimes], 0)

        yfull_test = p_test[:nsamples]
        thresh = np.mean(thresh, 0)
        result = (yfull_test > thresh)

        global inv_label_map
        preds = []
        for index in range(result.shape[0]):
            pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.where(result[index, :] == 1))))
            if len(pred) == 0:
                pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(yfull_test[index, :] - thresh))))

            preds.append(pred)

        df_test['tags'] = preds
        df_test.to_csv('../submission_keras_5_fold_CV_{}_LB_all.csv'.format(val_score), index=False)


    def test(self, thres, val_score, early_fusion=True, augmentation = 1, mirror=True):
        x_test = []
        df_test = pd.read_csv('../data/sample_submission_v2.csv')

        for f, tags in tqdm(df_test.values, miniters=1000):
            if f.startswith('test'):
                img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
                img = cv2.resize(img, (self.input_dim, self.input_dim))
                for angle in np.arange(0, augmentation*90, 90):
                    new_img = imutils.rotate(img, angle)
                    x_test.append(new_img)
                    if mirror:
                        x_test.append(cv2.flip(new_img, 1))
            else:
                img = cv2.imread('../data/test-jpg-additional/{}.jpg'.format(f))
                img = cv2.resize(img, (self.input_dim, self.input_dim))
                for angle in np.arange(0, augmentation*90, 90):
                    new_img = imutils.rotate(img, angle)
                    x_test.append(new_img)
                    if mirror:
                        x_test.append(cv2.flip(new_img, 1))
            # if(len(x_test) > 1000):
            #     break
        x_test  = np.array(x_test, np.float16)/255.0
        # for i in xrange(len(x_test)):
        #     x_test[i] -= mean

        ntimes = augmentation * (1 + mirror)
        nsamples = len(df_test)
        assert (x_test.shape[0] == ntimes * nsamples)
        yfull_test = []

        for i in xrange(self.nfolds):
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(i+1) + '.h5')
            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)
                p_test = self.model.predict(x_test, batch_size = self.batch_size, verbose=2)
                if ntimes > 1:
                    for j in xrange(nsamples):
                        p_test[j,:] = np.mean(p_test[j*ntimes:(j+1)*ntimes], 0)

                yfull_test.append(p_test[:nsamples])

        raw_result = np.zeros(yfull_test[0].shape)
        if early_fusion:
            thresh = np.zeros([1, len(thres[0])])
            for i in xrange(self.nfolds):
                raw_result += yfull_test[i]
                thresh += thres[i]

            print raw_result

            raw_result /= float(self.nfolds)
            thresh /= float(self.nfolds)
            result = (raw_result > thresh)
        else:
            for i in xrange(self.nfolds):
                raw_result += (yfull_test[i] > thres[i])
            result = raw_result / float(self.nfolds)

        global inv_label_map
        preds = []

        print thresh
        for index in range(result.shape[0]):
            pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.where(result[index, :] == 1))))
            if len(pred) == 0:
                if early_fusion:
                    print raw_result[index, :]
                    print thresh
                    pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(raw_result[index, :] - thresh))))
                else:
                    pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(raw_result[index, :]))))
            preds.append(pred)

        df_test['tags'] = preds
        df_test.to_csv('../submission_keras_5_fold_CV_{}_LB_.csv'.format(val_score), index=False)


    def test_new(self, thres, val_score, early_fusion=True):
        print('Testing on {} samples'.format(len(self.df_test_data)))

        def test_generator(transformation):
            while True:
                for start in range(0, len(self.df_test_data), self.batch_size):
                    x_batch = []
                    end = min(start + self.batch_size, len(self.df_test_data))
                    df_test_batch = self.df_test_data[start:end]
                    for f, tags in df_test_batch.values:
                        if f.startswith('test'):
                            img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
                        else:
                            img = cv2.imread('../data/test-jpg-additional/{}.jpg'.format(f))
                        img = cv2.resize(img, (self.input_dim, self.input_dim))
                        img = transformations(img, transformation)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32)
                    yield x_batch

        y_full_test = []
        for i in xrange(self.nfolds):
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(i + 1) + '.h5')
            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)

                # 6-fold TTA
                p_full_test = []
                for i in range(6):
                    p_test = self.model.predict_generator(generator=test_generator(transformation=i),
                                                     steps=math.ceil(len(self.df_test_data) / float(self.batch_size)))
                    p_full_test.append(p_test)

                p_test = np.array(p_full_test[0])
                for i in range(1, 6):
                    p_test += np.array(p_full_test[i])
                p_test /= 6
                y_full_test.append(p_test)

        raw_result = np.zeros(y_full_test[0].shape)
        if early_fusion:
            thresh = np.zeros([1, len(thres[0])])
            for i in xrange(self.nfolds):
                raw_result += y_full_test[i]
                thresh += thres[i]

            print raw_result

            raw_result /= float(self.nfolds)
            thresh /= float(self.nfolds)
            result = (raw_result > thresh)
        else:
            for i in xrange(self.nfolds):
                raw_result += (y_full_test[i] > thres[i])
            result = raw_result / float(self.nfolds)



        # final results
        preds = []

        print thresh
        for index in range(result.shape[0]):
            pred = ' '.join(list(map(lambda x: self.inv_label_map[x], *np.where(result[index, :] == 1))))
            if len(pred) == 0:
                if early_fusion:
                    pred = ' '.join(
                        list(map(lambda x: self.inv_label_map[x], *np.argmax(raw_result[index, :] - thresh))))
                else:
                    pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(raw_result[index, :]))))
            preds.append(pred)

        self.df_test_data['tags'] = preds
        self.df_test_data.to_csv('../submission_keras_5_fold_CV_{}_LB_.csv'.format(val_score), index=False)

if __name__ == "__main__":
    af = AmazonForest()

    thresh, val_score = af.train_new()
    # thresh, val_score = load_param()
    print("thresh:\n{}".format(thresh))
    print("val_score:", val_score)

    af.test_new(thres=thresh, val_score=val_score, early_fusion=True)

    # af.refine(thresh, val_score)
