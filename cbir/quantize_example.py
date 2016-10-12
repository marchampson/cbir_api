from __future__ import print_function
from mainwaring.ir import BagOfVisualWords
from sklearn.metrics import pairwise
import numpy as np

# rand gen vocabular/cluster centers along with the feature vectors, we'll generate 10 FV containing
# 6 real-valued entries, along with a codebook containing 3 'visual words'
np.random.seed(42)
vocab = np.random.uniform(size=(3,6))
features = np.random.uniform(size=(10,6))
print("[INFO] vocabulary:\n{}\n".format(vocab))
print("[INFO] features:\n{}\n".format(features))

# init bovw - it will contain 3 entries one for each of the possible visual words
hist = np.zeros((3,), dtype="int32")

# loop over the individual feature vectors
for (i, f) in enumerate(features):
    # compute euclidean dist between current fv and teh 3 visual words; then, find the index of the
    # word with teh smallest distance
    D = pairwise.euclidean_distances(f.reshape(1, -1),Y=vocab)
    j = np.argmin(D)

    print("[INFO] Closest visual word to feature #{}: {}".format(i, j))
    hist[j] += 1
    print("[INFO] Updated histogram: {}".format(hist))

#apply our BOVW class
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))