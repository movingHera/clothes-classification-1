import cv2 as cv
import os
import numpy as np

RULE_OUT_ATTR = []
def load_labels(label_filepath=r'../list_category_cloth.txt'):
    labels = []
    with open(label_filepath, 'r') as fl:
        fl.readline()
        fl.readline()
        for line in fl:
            labels.append(' '.join(line.strip().split()[:-1]))
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
        input_dir=r'G:\BaiduNetdiskDownload\DeepFashion\Category and Attribute Prediction Benchmark\Img',
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

    testLabelDistr = [0] * numCategory

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
                #break
                print(i)

            if numCategory == len(labels):
                # 1000-attr
                labels = np.where(np.array(list(map(int, labels))) > 0)[0].tolist()
                for l in labels:
                    labelDistr[l] += 1
            else:
                # 50-category
                labels = list(map(lambda x: int(x)-1, labels)) # for python 3, map(.) will return map object not list
                assert len(labels) == 1
                for l in labels:
                    labelDistr[l] += 1

            # convert labels to one-hot encoding
            targets = np.zeros(numCategory)
            for j in labels:
                targets[j] = 1

            if img_path not in partition_dict:
                y_train.append(targets)
                x_train.append(output_img_path)
            else:
                if partition_dict[img_path] == 'train':
                    y_train.append(targets)
                    x_train.append(output_img_path)
                elif partition_dict[img_path] == 'val':
                    y_val.append(targets)
                    x_val.append(output_img_path)
                elif partition_dict[img_path] == 'test':
                    for l in labels:
                        testLabelDistr[l] += 1

                    y_test.append(targets)
                    x_test.append(output_img_path)
    print(sum(labelDistr))
    # for j in range(numCategory):
    #     print('category {0}: {1}'.format(j+1, labelDistr[j]))
    label_map = {l: i for i, l in enumerate(classes)}
    inv_label_map = {i: l for l, i in label_map.items()}

    # save data distribution to .csv file
    keep_attr = save_data_dist(inv_label_map, labelDistr)
    keep_attr_ind = [label_map[i] for i in keep_attr]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = y_train[:, keep_attr_ind]
    y_val = y_val[:, keep_attr_ind]
    y_test = y_test[:, keep_attr_ind]
    numCategory = len(keep_attr_ind)

    x_train, y_train = remove_zero_annotation(x_train, y_train)
    x_val, y_val = remove_zero_annotation(x_val, y_val)
    x_test, y_test = remove_zero_annotation(x_test, y_test)
    print(keep_attr_ind)
    print(numCategory)

    return numCategory, label_map, inv_label_map, x_train, y_train, x_val, y_val, x_test, y_test

def save_data_dist(inv_label_map, labelDistr):
    distr = {inv_label_map[i]: j for i, j in enumerate(labelDistr)}
    result = sorted(distr.items(), lambda x, y: cmp(x[1], y[1])) # sort by values

    counts = [item[1] for item in result]
    attrs = [item[0] for item in result]
    import pandas as pd
    df = pd.DataFrame({'attributes': attrs, 'count': counts})
    df.to_csv('../attr_distr.csv', index=False)

    # rule out attributes with less than 5000 samples
    keep_attr = [item[0] for item in result if item[1] > 2000]
    return keep_attr

def remove_zero_annotation(x, y):
    n = len(x)
    ind = []
    for i in range(n):
        if y[i, :].max() > 0:
            ind.append(i)

    x = x[ind]
    y = y[ind,:]
    return x, y

if __name__ == '__main__':
    processing()
