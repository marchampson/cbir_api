from __future__ import print_function
from mainwaring.db import RedisQueue
from redis import Redis
import h5py

bovw_db = "/Users/marchampson/Desktop/cbir/output/bovw.hdf5"

# connect to redis, initialise the redis queue and open the bovw database
redisDB = Redis(host="localhost", port=6379, db=0)
redisDB.flushall()
rq = RedisQueue(redisDB)

bovwDB = h5py.File(bovw_db, mode="r")

# loop over the entries in the bag-of-visual-words
for (i, hist) in enumerate(bovwDB["bovw"]):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        print("[PROGRESS] processed {} entries".format(i))

    # add the image index and histogram to the redis server
    rq.add(i, hist)

# close the bag-of-visual-words database and finish the indexing processing
bovwDB.close()
rq.finish()