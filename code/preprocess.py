import cv2 as cv
import os
import numpy as np

def load_labels(label_filepath=r'../list_category_cloth.txt'):
    labels = []
    with open(label_filepath, 'r') as fl:
        fl.readline()
        fl.readline()
        for line in fl:
            labels.append(line.strip().split()[0])
    return labels

def load_bbox_dict(bbox_filepath='../list_bbox.txt'):
    bbox_dict = {}
    with open(bbox_filepath, 'r') as fl:
        fl.readline()
        fl.readline()
        for line in fl:
            line = line.strip().split()
            bbox_dict[line[0]] = map(lambda x: int(x), line[1:])
    return bbox_dict

def load_data_partition(partition_filepath='../list_eval_partition.txt'):
    partition_dict = {}
    with open(partition_filepath, 'r') as fl:
        fl.readline()
        fl.readline()
        for line in fl:
            line = line.strip().split()
            partition_dict[line[0]] = line[1]
    return partition_dict

def get_clothes_region(bbox_dict, input_dir, img_path):
    if img_path not in bbox_dict:
        return None
    input_img_path = os.path.join(input_dir, img_path)
    if not os.path.exists(input_img_path):
        return None
    x1, y1, x2, y2 = map(lambda x: int(x), bbox_dict[img_path])
    return cv.imread(input_img_path)[y1:y2, x1:x2]

def processing(
        input_dir=r'G:\BaiduNetdiskDownload\DeepFashion\Category and Attribute Prediction Benchmark\Img\img',
        output_dir=r'../input/crop_image',
        labels_dir=r'../list_category_cloth.txt',
        gt_dir=r'../list_category_img.txt'):
    if not os.path.exists(input_dir):
        return RuntimeError('Input directory does not exist.')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    classes = load_labels(labels_dir)
    numCategory = len(classes)
    labelDistr = [0] * numCategory

    bbox_dict = load_bbox_dict()
    partition_dict = load_data_partition()

    with open(gt_dir, 'r') as fl:
        fl.readline()
        fl.readline()
        i = 0
        for line in fl:
            img_path, labels = (lambda x: [x[0], x[1:]])(line.strip().split())

            # print(img_path)

            output_img_path = os.path.join(output_dir, img_path)
            if not os.path.exists(output_img_path):
                output_subdir = output_img_path[:output_img_path.rfind('/')]
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                crop_image = get_clothes_region(bbox_dict, input_dir, img_path)
                if crop_image is None:
                    continue
                cv.imwrite(output_img_path, crop_image)

            i += 1
            if i % 10000 == 0:
                print(i)

            if numCategory == len(labels):
                labels = np.where(np.array(list(map(int, labels))) > 0)[0].tolist()
                for l in labels:
                    labelDistr[l] += 1
            else:
                labels = list(map(lambda x: int(x)-1, labels)) # for python 3, map(.) will return map object not list
                for l in labels:
                    labelDistr[l] += 1

            if img_path not in partition_dict:
                y_train.append(labels)
                x_train.append(output_img_path)
            else:
                if partition_dict[img_path] == 'train':
                    y_train.append(labels)
                    x_train.append(output_img_path)
                elif partition_dict[img_path] == 'val':
                    y_val.append(labels)
                    x_val.append(output_img_path)
                elif partition_dict[img_path] == 'test':
                    y_test.append(labels)
                    x_test.append(output_img_path)
    print(sum(labelDistr))
    for j in range(numCategory):
        print('category {0}: {1}'.format(j+1, labelDistr[j]))

    return numCategory, x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == '__main__':
    processing()