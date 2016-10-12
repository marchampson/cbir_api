from __future__ import print_function
# More scalable and efficient version of k-means algorithm
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py

class Vocabulary:
    def __init__(self, dbPath, verbose=True):
        # store params as vars
        self.dbPath = dbPath
        self.verbose = verbose

    def fit(self, numClusters, samplePercent, randomState=None):
        # open the db and get the total number of features
        db = h5py.File(self.dbPath)
        totalFeatures = db["features"].shape[0]

        print("Total features {}".format(totalFeatures))

        # determine the number of features to sample, generate the indexes of the sample,
        # sorting them in ascending order to speedup access time from the HDF5 database
        sampleSize = int(np.ceil(samplePercent * totalFeatures))
        print("sampleSize {}".format(sampleSize))

        if(numClusters == 0):
            numClusters = sampleSize / 4

        print("numClusters: {}".format(numClusters))

        idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling...")

        # loop over the randomly sampled indexes and accumulate the features to cluster
        for i in idxs:
            data.append(db["features"][i][2:])

        # cluster the data
        self._debug("sampled {:,} features from a population of {:,}".format(len(idxs), totalFeatures))
        self._debug("clustering with k={:,}".format(numClusters))
        print("n_clusters {} random_state {}".format(numClusters, randomState))
        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        #print("random_state {}".format(randomState))
        clt.fit(data)
        self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

        # close the database
        db.close()

        # return the cluster centroids
        return clt.cluster_centers_

    def _debug(self, msg, msgType="[INFO]"):
        # check to see if the message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))