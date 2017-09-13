# -*- coding: utf-8 -*-

from keras.models import model_from_json
import cv2
import numpy as np

import time
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



global model
class TF_HeadSeg():
    def __init__(self, model_file, weight_file):
        self.model_file = model_file
        self.weight_file = weight_file
        self.seg_model = None
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
            self.seg_model = model_from_json(loaded_model_json)
            # print self.seg_model
            # print os.path.isfile(self.weight_file)
            if not os.path.exists(self.weight_file):
                raise RuntimeError('No weights file founded.')
            self.seg_model.load_weights(self.weight_file)

            self.graph = tf.get_default_graph()


    def inference(self, image):
        '''
        predict the mask given specific image and the model object
        :param image: [1, 512, 512, 3] 'float32' input image normalized to [0, 1]
        :param model: model object
        :return: [1, 512, 512, 1] 'float32' output mask
        '''
        # print image.shape
        # print image.dtype

        m_Height, m_Width = image.shape[:2]
        img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, *img.shape)

        with self.graph.as_default():

            mask = self.seg_model.predict(img, batch_size=1)[0]
            prob = cv2.resize(mask[0], (m_Width, m_Height))
            mask = prob > 0.5

            # 连通区域****************************************************
            mask_copy = mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_copy, 0, 2)
            contourSize = len(contours)
            if contourSize == 0:
                return 0
            areas = np.zeros(contourSize)
            idx = 0
            maxArea = -1.0
            for cont in contours:
                area = cv2.contourArea(cont)
                areas[idx] = area
                idx = idx + 1
                if maxArea < area:
                    maxArea = area

            # 必须要满足这个条件才，也就是满足15分之一原图像面积大小的才是人脸区域
            srcArea = m_Width * m_Height
            if maxArea < 0.066 * srcArea:
                return 0
            img_draw = np.zeros(mask.shape, dtype=np.uint8)
            for idx in range(contourSize):
                if areas[idx] >= maxArea:
                    cv2.drawContours(img_draw, contours, idx, [255], -1)
                else:
                    cv2.drawContours(img_draw, contours, idx, [0], -1)

            # 向外扩充5个像素
            erodeSize = 5
            kernel = np.uint8(np.zeros((erodeSize, erodeSize)))
            for x in range(5):
                kernel[x, 2] = 1;
                kernel[2, x] = 1;
            dilate_img = cv2.dilate(img_draw, kernel)  # 膨胀图像
            res_img = np.zeros(mask.shape, dtype=np.uint8)

            mask_res = np.zeros(mask.shape, dtype=np.uint8)
            mask_res = cv2.absdiff(dilate_img, img_draw);
            mask_res = mask_res * prob
            mask_res = mask_res.astype(np.uint8)
            mask_res += img_draw

            #res_img = img_draw
            r, g, b = cv2.split(image)
            result = cv2.merge([r, g, b, mask_res])

            #cv2.imwrite("result.png", result)
            # end = time.clock()
            # print("后处理用时: %f s" % (end - start))
            # print("总的运行用时: %f s" % (end - start1))

        return result

# def head_seg(img_vec):
#     #print("start to seg")
#     #print(len(img_vec))
#     raw_img = np.array(img_vec)
#     #print(raw_img.shape)
#     raw_img = raw_img.reshape(800, 600,3).astype(np.uint8)
#     # print(raw_img.shape)
#     # print(type(raw_img))
#     # print(raw_img.dtype)
#     # raw_img = cv2.imread(image)
#     #H, W = raw_img.shape[:2]
#
#     img = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_LINEAR)
#     img = img.astype(np.float32) / 255.0
#     img = img.reshape(1, *img.shape)
#
#     mask = seg_model.predict(img, batch_size=1)[0]
#     #return seg_model.predict(img, batch_size=1)
#
#     return mask.flatten().tolist()
# if __name__ == '__main__':
#     init('headseg_net.json', 'headseg_weights.h5')
#     image  = cv2.imread('test.jpg')
#     pred = head_seg(image)
#     start0 = time.clock()
#
#     pred = head_seg(image)
#
#     end0 = time.clock()
#     print("time: %f s" % (end0 - start0))
#
#     print(len(pred))
#     print(max(pred))
#     print(min(pred))
#     #print(pred)
#     # print pred.shape
#     # print pred.dtype
#
#     #W = 600
#     #H = 800
#     #prob = cv2.resize(pred[0], (W, H))
#     #mask = prob > 0.5
#     #mask = np.dstack((mask,) * 3).astype(np.uint8)
#     #cv2.imwrite('tmp_res.png', mask*255)
#
#