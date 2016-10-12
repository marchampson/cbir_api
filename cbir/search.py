from __future__ import print_function
from mainwaring.descriptors import DetectAndDescribe
from mainwaring.descriptors import Kaze
from mainwaring.ir import BagOfVisualWords
from mainwaring.ir import Searcher
from mainwaring.ir import dists
from scipy.spatial import distance
from redis import Redis
import cPickle
import imutils
import json
import cv2

# initialize the keypoint detector, local invariant descriptor, descriptor pipeline,
# distance metric, and inverted document frequency array
detector = cv2.KAZE_create()
descriptor = Kaze()
dad = DetectAndDescribe(detector, descriptor)

idf = "/Users/marchampson/Desktop/cbir/output/idf.cpickle"
codebook = "/Users/marchampson/Desktop/cbir/output/vocab.cpickle"
query = "/Users/marchampson/Desktop/cbir/queries/10.jpg"
bovw_db = "/Users/marchampson/Desktop/cbir/output/bovw.hdf5"
features_db = "/Users/marchampson/Desktop/cbir/output/features.hdf5"

#idf = None
#distanceMetric = dists.chi2_distance

# if the path to the inverted document frequency array was supplied, then load the
# idf array and update the distance metric
#if args["idf"] is not None:
idf = cPickle.loads(open(idf).read())
distanceMetric = distance.cosine

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(codebook).read())
bovw = BagOfVisualWords(vocab)

# load the relevant queries dictionary and look up the relevant results for the
# query image
#relevant = json.loads(open(args["relevant"]).read())
#queryFilename = args["query"][args["query"].rfind("/") + 1:]
#queryRelevant = relevant[queryFilename]

# load the query image and process it
queryImage = cv2.imread(query)
cv2.imshow("Query", imutils.resize(queryImage, width=320))
cv2.waitKey(0)
queryImage = imutils.resize(queryImage, width=320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# extract features from the query image and construct a bag-of-visual-words from it
(_, descs) = dad.describe(queryImage)
hist = bovw.describe(descs).tocoo()

# connect to redis and perform the search
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, bovw_db, features_db, idf=idf,
                    distanceMetric=distanceMetric)
sr = searcher.search(hist, numResults=2, maxCandidates=2)
print("[INFO] search took: {:.2f}s".format(sr.search_time))

# initialize the results montage
#montage = ResultsMontage((240, 320), 5, 20)

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(sr.results):
    # load the result image and display it

    print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num=i + 1,
                                                                 result=resultID, score=score))
    #result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    #montage.addResult(result, text="#{}".format(i + 1),
    #                  highlight=resultID in queryRelevant)

# show the output image of results
#cv2.imshow("Results", imutils.resize(montage.montage, height=700))
#cv2.waitKey(0)
searcher.finish()