
import numpy as np
import time
from numba import jit
from numba import cuda
import Util




@cuda.jit
def GSSE_SP(timeSeries, indexes, nb_timeSeries, k, sse):
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    i = tx + ty * bw
    #print("i=",i," k=",k)
    if i!=0 and i < k:
        #indexes_temp = np.delete(indexes,i)
        #print("i=",i," k=",k,"len ",length," ",length_indexes)
        indexes_temp = cuda.local.array(shape=1350, dtype=np.int16)
        for l in range(i):
            indexes_temp[l]=indexes[l]
        for l in range(i,len(indexes)-1):
            indexes_temp[l]=indexes[l+1]

        sum=0
        for j in range(nb_timeSeries):
            # paa
            ts_PAA = cuda.local.array(shape=1349, dtype=np.float64)
            for l in range(len(indexes)-2):
                ts_PAA[l] = Util.seg_mean(timeSeries[j], indexes_temp[l], indexes_temp[l + 1] - 1)

            m = 0
            for ind in range(len(indexes)-2):
                for l in range(indexes_temp[ind], indexes_temp[ind + 1]):
                    m += (ts_PAA[ind] - timeSeries[j][l]) ** 2
            sum = sum + m

        sse[i] =sum

def PASAX_GSSE_SP(timeSeries,nb_segments,segments_size_start):
    nb_timeSeries = timeSeries.shape[0]
    ts_len = timeSeries.shape[1]
    nb_segments_start = ts_len // segments_size_start


    # print("nb seg start",nb_segments_start)
    indexes = np.empty(nb_segments_start + 1)
    indexes[0] = 0
    for j in range(1, nb_segments_start):
        indexes[j] = indexes[j - 1] + segments_size_start
    indexes[nb_segments_start] = ts_len

    k = nb_segments_start


    while(k!=nb_segments):


        sse = cuda.device_array(k)
        sse[0] = np.inf

        threadsperblock = 32
        blockspergrid = (k + (threadsperblock - 1)) // threadsperblock


        GSSE_SP[blockspergrid, threadsperblock](timeSeries, indexes, nb_timeSeries, k, sse)
        #print(mse)
        ssea=sse.copy_to_host()
        indexToMergePosition = np.argmin(ssea)
        #print(indexToMergePosition)
        indexes = np.delete(indexes, indexToMergePosition)

        k = k - 1
        #print("fin")

    return indexes

# Test

#fileName="HandOutlines.txt"
fileName="ECGFiveDays.txt"
data,m=Util.readDataset(fileName)

timeSeries=cuda.to_device(data)

nb_segment = m // 10
min=float('inf')
for i in range(5):
    start=time.time()
    PASAX_GSSE_SP(timeSeries, nb_segment, 2)
    end=time.time()
    exec_time=end - start
    if exec_time<min:
        min=exec_time
print("Elapsed = ", min)



