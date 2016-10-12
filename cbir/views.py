from __future__ import print_function
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
import numpy as numpy
import urllib
import json
import cv2
import os

# image matching
import cPickle
import imutils

from mainwaring.descriptors import DetectAndDescribe
from mainwaring.descriptors import Kaze
from mainwaring.ir import BagOfVisualWords
from mainwaring.ir import Searcher
from mainwaring.ir import dists
from scipy.spatial import distance
from redis import Redis

@csrf_exempt
def detect(request):
	SITE_ROOT = os.path.dirname(os.path.realpath(__file__))

	# Initialise the data dictionary
	data = {"success": False}

	if request.method == "POST":

		image = _grab_image("/Users/marchampson/Desktop/cbir_api/cbir/queries/9.jpg")

		detector = cv2.KAZE_create()
		descriptor = Kaze()
		dad = DetectAndDescribe(detector, descriptor)

		idf = cPickle.loads(open("/Users/marchampson/Desktop/cbir_api/cbir/output/idf.cpickle").read())
        distanceMetric = distance.cosine

        # load the codebook vocabulary and initialize the bag-of-visual-words transformer
        vocab = cPickle.loads(open("/Users/marchampson/Desktop/cbir_api/cbir/output/vocab.cpickle").read())
        bovw = BagOfVisualWords(vocab)

        queryImage = image
        queryImage = imutils.resize(queryImage, width=320)
        queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

        # extract features from the query image and construct a bag-of-visual-words from it
        (_, descs) = dad.describe(queryImage)
        hist = bovw.describe(descs).tocoo()

        # connect to redis and perform the search
        redisDB = Redis(host="localhost", port=6379, db=0)
        searcher = Searcher(redisDB, "/Users/marchampson/Desktop/cbir_api/cbir/output/bovw.hdf5",
        	"/Users/marchampson/Desktop/cbir_api/cbir/output/features.hdf5", idf=idf, distanceMetric=distanceMetric)
        sr = searcher.search(hist, numResults=1, maxCandidates=2)
        print("[INFO] search took: {:.2f}s".format(sr.search_time))

        for (i, (score, resultID, resultIdx)) in enumerate(sr.results):
    		print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num=i + 1, result=resultID, score=score))

        searcher.finish()

        data.update({"results": resultID, "success": True})

	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()

        # if the steram is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

@require_http_methods(["POST"])
@csrf_exempt
def add(request):
    # Only GET and POST methods can make it this far
    print("hello")

    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded

        if request.FILES.get("image", None) is not None:
            # print(request.FILES.get("image"))
            # grab the uploaded image
            #image = _grab_image(stream=request.FILES["image"])
            image = request.FILES["image"]

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            # if url is None:
            #    data["error"] = "No URL provided."
            #    return JsonResponse(data)

            # load the image and convert
            # image = _grab_image(url=url)

            return JsonResponse(data)

    handle_uploaded_file(image)

    # update the data dictionary with the restuls
    data.update({"results": "foo", "success": True})

    #  return a JSON response
    return JsonResponse(data)

def handle_uploaded_file(f):
    filename = f.name
    with open('/Users/marchampson/Desktop/testupload/' + filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
