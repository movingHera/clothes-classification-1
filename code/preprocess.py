import cv2 as cv
import os

def load_labels(label_file_path=r'../list_category_cloth.txt'):
    labels = []
    with open(label_file_path, 'r') as fl:
        fl.readline()
        fl.readline()
        for line in fl:
            labels.append(line.strip().split()[0])
    return labels

def processing(input_dir=r'G:\BaiduNetdiskDownload\DeepFashion\Category and Attribute Prediction Benchmark\Img\img',
               output_dir=r'../input/crop_image'):
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
    numCategory = 50
    labelDistr = [0] * numCategory

    def load_bbox_dict():
        bbox_dict = {}
        with open('../list_bbox.txt', 'r') as fl:
            fl.readline()
            fl.readline()
            for line in fl:
                line = line.strip().split()
                bbox_dict[line[0]] = map(lambda x: int(x), line[1:])
        return bbox_dict

    def load_data_partition():
        partition_dict = {}
        with open('../list_eval_partition.txt', 'r') as fl:
            fl.readline()
            fl.readline()
            for line in fl:
                line = line.strip().split()
                partition_dict[line[0]] = line[1]
        return partition_dict

    def get_clothes_region(bbox_dict, img_path):
        if img_path not in bbox_dict:
            return None
        input_img_path = os.path.join(input_dir, img_path)
        if not os.path.exists(input_img_path):
            return None
        x1, y1, x2, y2 = map(lambda x: int(x), bbox_dict[img_path])
        return cv.imread(input_img_path)[y1:y2, x1:x2]

    bbox_dict = load_bbox_dict()
    partition_dict = load_data_partition()
    with open('../list_category_img.txt', 'r') as fl:
        fl.readline()
        fl.readline()
        i = 0
        for line in fl:
            img_path, label = line.strip().split()

            # print(img_path)

            crop_image = get_clothes_region(bbox_dict, img_path)
            if crop_image is None:
                continue
            i += 1
            if i % 10000 == 0:
                print(i)
            label = int(label)-1
            output_img_path = os.path.join(output_dir, img_path)
            if not os.path.exists(output_img_path):
                output_subdir = output_img_path[:output_img_path.rfind('/')]
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                cv.imwrite(output_img_path, crop_image)
            labelDistr[label-1] += 1

            if img_path not in partition_dict:
                y_train.append(label)
                x_train.append(output_img_path)
            else:
                if partition_dict[img_path] == 'train':
                    y_train.append(label)
                    x_train.append(output_img_path)
                elif partition_dict[img_path] == 'val':
                    y_val.append(label)
                    x_val.append(output_img_path)
                elif partition_dict[img_path] == 'test':
                    y_test.append(label)
                    x_test.append(output_img_path)
    print(sum(labelDistr))
    for i in range(numCategory):
        print('category {0}: {1}'.format(i+1, labelDistr[i]))

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == '__main__':
    processing()