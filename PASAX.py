import numpy as np
import Util
from numba import jit
import time


@jit(nopython=True)
def PASAX_GSSE(timeSeries, nb_segments, segments_size_start):
    nb_timeSeries=timeSeries.shape[0]
    ts_len=timeSeries.shape[1]
    nb_segments_start=ts_len//segments_size_start
    #print("nb seg start",nb_segments_start)
    indexes=np.empty(nb_segments_start+1)
    indexes[0]=0
    for j in range(1,nb_segments_start):
        indexes[j]=indexes[j-1]+segments_size_start
    indexes[nb_segments_start]=ts_len
    k = nb_segments_start
    while(k!=nb_segments):
        #print("k=",k)
        #print("****")
        #print("indexes",indexes)
        #print("****")
        indexToMergePosition=0
        sse=np.inf
        for i in range(1,k):
            #print("i=",i)
            #print("index len ", len(indexes))
            #print("i=", i)
            indexes_temp = np.delete(indexes,i)

            #print("index temp",indexes_temp)
            sse_array=np.zeros(nb_timeSeries)
            for j in range(nb_timeSeries):
                #paa
                ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
                #print(ts_PAA)

                sum=0
                for ind in range(len(indexes_temp)-1):
                    for l in range(indexes_temp[ind],indexes_temp[ind+1]):
                        sum+=(ts_PAA[ind]-timeSeries[j][l])**2

                #print("sum = ",sum)
                sse_array[j]=sum
            #print("sse array :",sse_array)

            sseTemp=np.sum(sse_array)

            if(sseTemp<sse):
                sse=sseTemp
                indexToMergePosition=i



        indexes=np.delete(indexes,indexToMergePosition)
        k=k-1
        #print("fin")

    return indexes

@jit(nopython=True)
def PASAX_LSSE(timeSeries, nb_segments, segments_size_start):
    nb_timeSeries=timeSeries.shape[0]
    ts_len=timeSeries.shape[1]
    nb_segments_start=ts_len//segments_size_start
    #print("nb seg start",nb_segments_start)
    indexes=np.empty(nb_segments_start+1)
    indexes[0]=0
    for j in range(1,nb_segments_start):
        indexes[j]=indexes[j-1]+segments_size_start
    indexes[nb_segments_start]=ts_len


    k = nb_segments_start
    while(k!=nb_segments):
        #print("k=",k)
        #print("indexes",indexes)
        indexToMergePosition=0
        sse=np.inf

        for i in range(1,k):
            #print("i=",i)
            #print("index len ", len(indexes))
            indexes_temp = np.array((indexes[i-1],indexes[i+1]))
            sse_array = np.zeros(nb_timeSeries)
            for j in range(nb_timeSeries):
                #paa
                ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
                sum=0
                for l in range(indexes[i-1],indexes[i+1]):
                    sum+=(ts_PAA[0]-timeSeries[j][l])**2

                sse_array[j] = sum

            sseTemp = np.sum(sse_array)

            if(sseTemp<sse):
                sse=sseTemp

                indexToMergePosition=i


        indexes=np.delete(indexes,indexToMergePosition)
        #print('itmp ',indexToMergePosition)
        k=k-1

    return indexes

