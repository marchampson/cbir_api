# import packages
import numpy as np
import cv2

class Kaze:
    def __init__(self):
        # init the KAZE feature extractor
        self.extractor = cv2.KAZE_create()

    def compute(self, image, kps, eps=1e-7):
        # compute descriptors
        (kps, descs) = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # return a tuple of the keypoints and descriptors
        return (kps, descs)