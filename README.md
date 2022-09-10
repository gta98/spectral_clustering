# Spectral Clustering

## What is this?

This is my final assignment in a course I took, "Software Project" (0368.2161)

The code performs [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering) on any given comma-delimited csv dataset - detailed documentation of the algorithm itself can be found in `./tests/resources/sp_project.pdf`, which was provided by course staff

## Omg wowa wiwa! How did you do this?

- I reused my previous work on [kmeans++](https://github.com/gta98/kmeansplusplus)

- I significantly optimized some parts of the algorithm by hardcoding matrix multiplication on rotation matrices, which are very close in our case to the identity matrix -  this greatly reduced computational complexity in Jacobi iterations

- Most of the "heavy lifting" is done in C, with Python hooks in `./spkmeansmodule.c`

- You can find a bunch of unit tests in `./tests`

## But are there any limitations?

- The data is loaded straight to memory, however it should be trivial, by modifying `matrix.c`, to allow for (slow) caching and even data manipulation on persistent storage, if we're dealing with extremely large amounts of data (i.e. many datapoints / large dimensionality)

- Yes! Many more
