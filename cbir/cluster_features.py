from __future__ import print_function
from mainwaring.ir import Vocabulary
import cPickle

codebook = '/Users/marchampson/Desktop/cbir/output/vocab.cpickle'

# create the visual words vocabulary
voc = Vocabulary('/Users/marchampson/Desktop/cbir/output/features.hdf5')

# vocab params clusters = 1500, percentage = 0.25
# set clusters to 0 to auto calculate
vocab = voc.fit(0, 0.25)

# dump the clusters to file
print("[INFO] storing cluster centers...")
f = open(codebook, "w")
f.write(cPickle.dumps(vocab))
f.close()