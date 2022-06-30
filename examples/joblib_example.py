import time
import numpy as np
from math import sqrt
from joblib import Parallel, delayed

start = time.time()
res = [sqrt(i**2) for i in range(10)]
print("Python list: {} seconds".format(time.time() - start))

start = time.time()
res = np.sqrt(np.square(np.arange(10)))
print("Numpy array: {} seconds".format(time.time() - start))

start = time.time()
res = Parallel(n_jobs=-1)(delayed(sqrt)(i**2) for i in range(10))
print("joblib: {} seconds".format(time.time() - start))
