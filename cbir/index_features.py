from __future__ import print_function
from mainwaring.descriptors import DetectAndDescribe
from mainwaring.descriptors import Kaze
from mainwaring.indexer import FeatureIndexer
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np

# initialise the keypoint detector, local invariant descriptor and the method
detector = cv2.KAZE_create()
descriptor = Kaze()
dad = DetectAndDescribe(detector, descriptor)

features_db = "/Users/marchampson/Desktop/cbir/output/features.hdf5"

# initialise the feature indexer
fi = FeatureIndexer(features_db, estNumImages=10, maxBufferSize=50000, verbose=True)

# We're running this as an api some command line arguments are fixed
dataset = '/Users/marchampson/Desktop/cbir/covers'

# loop over the images in the dataset
for (i, imagePath) in enumerate(paths.list_images(dataset)):
    # Show progress every 10
    if i > 0 and i % 10  == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # extract the image filename (unique image ID) from the path then load image
    filename = imagePath[imagePath.rfind("/") + 1:]
    print(filename)
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # describe the image
    (kps, descs) = dad.describe(image)

    # if either kps or descs are None, ignore
    if kps is None or descs is None:
        continue

    print("# of keypoints: {}".format(len(kps)))

    fi.add(filename, kps, descs)

    # loop over the keypoints and draw them
    # kps = np.int0([kp.pt for kp in kps])

    for kp in kps:
        r = int(0.5 * 2)
        (x, y) = kp
        cv2.circle(copy, (x,y), r, (0, 255, 255), 1)

    cv2.imshow("Image", copy)
    cv2.waitKey(0)


# finish the indexing process
fi.finish()