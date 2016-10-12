from mainwaring.ir import BagOfVisualWords
from mainwaring.indexer import BOVWIndexer
import cPickle
import h5py

codebook = "/Users/marchampson/Desktop/cbir/output/vocab.cpickle"
features_db = "/Users/marchampson/Desktop/cbir/output/features.hdf5"
bovw_db = "/Users/marchampson/Desktop/cbir/output/bovw.hdf5"
max_buffer_size = 500
idf = "/Users/marchampson/Desktop/cbir/output/idf.cpickle"

# load the codebook vocab and initialise the bag-of-visual-words tranformer
vocab = cPickle.loads(open(codebook).read())
bovw = BagOfVisualWords(vocab)

# open the features database and initialise the bovw indexer
featuresDB = h5py.File(features_db, mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], bovw_db,
                 estNumImages=featuresDB["image_ids"].shape[0],
                 maxBufferSize=max_buffer_size)


# loop over the image IDs and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	hist = bovw.describe(features)

	# add the bag-of-visual-words to the index
	bi.add(hist)

# close the features database and finish the indexing process
featuresDB.close()
bi.finish()

# dump the inverse document frequency counts to file
f = open(idf, "w")
f.write(cPickle.dumps(bi.df(method="idf")))
f.close()