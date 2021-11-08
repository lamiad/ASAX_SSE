import numpy as np
import time
import Util
from numba import cuda





@cuda.jit
def LSSE_DP(timeSeries, nb_threads, indexes_temp, sum_sse):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos < nb_threads:
        # print("i=",i)

        ts_PAA = Util.seg_mean(timeSeries[pos], indexes_temp[0], indexes_temp[1] - 1)
        sum = 0
        for l in range(indexes_temp[0], indexes_temp[1]):
            sum += (ts_PAA- timeSeries[pos][l]) ** 2

        sum_sse[pos] = sum


def PASAX_LSSE_DP(timeSeries, nb_segments, segments_size_start, nb_threads):

    ts_len = timeSeries.shape[1]
    nb_segments_start = ts_len // segments_size_start
    indexes = np.empty(nb_segments_start + 1)
    indexes[0] = 0
    for j in range(1, nb_segments_start):
        indexes[j] = indexes[j - 1] + segments_size_start
    indexes[nb_segments_start] = ts_len
    k = nb_segments_start

    while (k != nb_segments):
        # print("****")
        # print("indexes",indexes)
        indexToMergePosition = 0
        sse = np.inf
        for i in range(1, k):

            indexes_tempb = np.array((indexes[i - 1], indexes[i + 1]))
            indexes_temp = cuda.to_device(indexes_tempb)
            sum_sse = cuda.device_array(nb_threads)
            threadsperblock = 32
            blockspergrid = (nb_threads + (threadsperblock - 1)) // threadsperblock

            LSSE_DP[blockspergrid, threadsperblock](timeSeries, nb_threads, indexes_temp, sum_sse)

            sum_ssea = sum_sse.copy_to_host()

            sseTemp = np.sum(sum_ssea)

            if (sseTemp < sse):
                sse = sseTemp
                indexToMergePosition = i

        indexes = np.delete(indexes, indexToMergePosition)

        k = k - 1

    return indexes

# Test

fileName="ECGFiveDays.txt"
data,m=Util.readDataset(fileName)

timeSeries = cuda.to_device(data)

nb_segment=m//10

nb_threads = timeSeries.shape[0]

min = float('inf')

for i in range(5):
    start = time.time()
    PASAX_LSSE_DP(timeSeries, nb_segment, 2, nb_threads)
    end = time.time()
    exec_time = end - start
    if exec_time < min:
        min = exec_time

print("Elapsed = ", min)


