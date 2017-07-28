from sklearn.metrics import fbeta_score
import numpy as np
import cv2
import imutils

def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

def load_param():

    thresh = [[0.03, 0.03, 0.05, 0.07, 0.03, 0.02, 0.05, 0.03, 0.05, 0.05, 0.04, 0.03, 0.05, 0.1, 0.04, 0.04, 0.06],
     [0.05, 0.03, 0.09, 0.08, 0.03, 0.02, 0.05, 0.08, 0.04, 0.05, 0.02, 0.03, 0.03, 0.07, 0.04, 0.06, 0.05],
     [0.04, 0.03, 0.05, 0.06, 0.02, 0.04, 0.05, 0.05, 0.03, 0.05, 0.04, 0.03, 0.03, 0.11, 0.04, 0.05, 0.06],
     [0.02, 0.03, 0.06, 0.1, 0.03, 0.01, 0.06, 0.05, 0.04, 0.1, 0.05, 0.03, 0.03, 0.1, 0.04, 0.05, 0.1],
     [0.04, 0.03, 0.04, 0.06, 0.03, 0.03, 0.05, 0.09, 0.03, 0.07, 0.07, 0.04, 0.04, 0.08, 0.03, 0.06, 0.09]]
    val_score = 0.93065441478548683

    return thresh, val_score

def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.21):
    _, num_classes = labels.shape[0:2]
    best_thresholds = [seed] * num_classes
    best_scores = [0] * num_classes
    for t in range(num_classes):

        thresholds = list(best_thresholds)  # [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f' % (t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores


def normallize(img):
    img = img.astype(np.float16)

    img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
    img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
    img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017
    img = np.expand_dims(img, axis=0)
    return img


def transformations(src, choice):
    if choice == 0:
        # Rotate 90
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    return src

def transformations2(src, choice):
    mode = choice // 2
    src = imutils.rotate(src, mode * 90)
    if choice % 2 == 1:
        src = cv2.flip(src, flipCode=1)
    return src
