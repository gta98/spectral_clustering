This project is an assignment I was given in one of my courses

It performs Kmeans clustering on a given dataset (csv/txt) using the Kmeans++ initialization method, using Numpy, CPython, and testing against Sklearn

I implemented the algorithm in bare C, and wrote a wrapper in CPython for easy interfacing with Python - see `source/kmeans.c`

The initialization (Kmeans++) is performed in `source/kmeans_pp.py`, which is also the main thread of the program - in `KMeansPlusPlus()`.

For the initialization part, there is also a more elegant solution which randomizes in a way that is slightly different than the one provided in the example outputs, so I ended up not using it. See `KMeansPlusPlus_original()`

I also wrote some pretty extensive tests - among other things, comparing the results to sklearn, in `source/tests.py`

