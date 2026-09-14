"""Unchanged source DFDC five-landmark removal operations.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. These use the five-point preprocessing landmarks,
separate from the 68-point predictor required by convex-hull augmentation.
Source zero dilation iterations mean repeat-until-stable, not no dilation.
"""

import math
import random
import cv2
import numpy as np
from scipy.ndimage import binary_dilation

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image

def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image

def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image

def remove_landmark(image, landmarks):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image
