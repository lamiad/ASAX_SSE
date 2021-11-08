from numba import jit
import numpy as np
import csv
import random



def normalization(ts):
    ts_mean=np.mean(ts)
    #print(ts_mean)
    ts_std=np.std(ts)
    #print(ts_std)
    ts_normalized=np.empty_like(ts)
    #print(ts_normalized)
    for i in range(len(ts)):
        ts_normalized[i]=(ts[i]-ts_mean)/ts_std
    return ts_normalized

def ds_normalization(timesSeries):
    timeSeries_normalized = np.empty((timesSeries.shape[0], timesSeries.shape[1]))
    i = 0
    for ts in timesSeries:
        timeSeries_normalized[i] = normalization(ts)
        i += 1
    return timeSeries_normalized
@jit(nopython=True)
def seg_mean(ts,start,end):
    sum=0
    #print(start," hta ",end)
    for i in range(start,end+1):
        sum+=ts[i]
    #print("end =",end," start=",start)
    #print(sum/(end-start+1))
    return sum/(end-start+1)


def eucDistance(q,c):
    return np.sqrt(np.sum((q-c)**2))

def DR(q_PAA,c_PAA,ts_len):
    return np.sqrt(ts_len/len(q_PAA))*np.sqrt(np.sum((q_PAA-c_PAA)**2))

def DR_VAR(q_PAA,c_PAA,segs_len):
    sum = 0
    for i in range(len(q_PAA)):
        sum += ((q_PAA[i]-c_PAA[i]) ** 2) * (segs_len[i])

    return np.sqrt(sum)

def MINDIST(q_s,c_s,ts_len,cuts):
    return np.sqrt(ts_len/len(q_s))*np.sqrt(sum_dist(q_s,c_s,cuts))

def sum_dist(q_s,c_s,cuts):
    sum=0
    for i in range(len(q_s)):
        sum+=(dist(q_s[i],c_s[i],cuts))**2
    return sum

def dist(r,c,cuts):
    r=int(r)
    c=int(c)
    if abs(r-c)<=1:
        return 0
    else:
        return cuts[max(r,c)-1]-cuts[min(r,c)]

def MINDIST_VAR(q_s,c_s,segs_len,cuts):
    return np.sqrt(sum_dist_var(q_s,c_s,segs_len,cuts))

def sum_dist_var(q_s,c_s,segs_len,cuts):
    sum=0
    for i in range(len(q_s)):
        sum+=((dist(q_s[i],c_s[i],cuts))**2)*(segs_len[i])
    return sum

def segments_len(indexes):
    w=len(indexes)-1
    segs_len=np.empty(w,dtype=int)
    for i in range(w):
        segs_len[i]=indexes[i+1]-indexes[i]
    return segs_len


def extractMin(pq):
    min_val=float('inf')
    key=None
    for k,val in pq.items():
        if val<min_val:
            key=k
    return key

def PAA_fixedSegSize(ts,nb_segments):

    ts_len=len(ts)
    segment_size=ts_len/nb_segments
    ts_PAA=np.empty(nb_segments)
    if nb_segments!=ts_len:

        offset = 0
        for i in range(nb_segments):
            ts_PAA[i]=seg_mean(ts,offset,offset+int(segment_size)-1)
            #print("paa i ",ts_PAA[i])
            offset+=int(segment_size)
    else:
        ts_PAA=np.copy(ts)

    return ts_PAA
@jit(nopython=True)
def PAA_varSegSize(ts,indexes):

    nb_segments=len(indexes)-1
    ts_PAA = np.empty(nb_segments)

    for i in range(nb_segments):
        ts_PAA[i]=seg_mean(ts,indexes[i],indexes[i+1]-1)

    return ts_PAA


def readDataset(file):

    with open("Data/" + file) as f:
        for line in f:
            pass
        last_line = line

    v=last_line.split(" ")
    n=int(v[0])
    m=int(v[1])

    file = open("Data/" + file, "r")
    timeSeries = np.empty((n, m))
    for i in range(n):
        ts_StrValues = file.readline().split(",")
        # print(ts_StrValues)
        # print(i,"   ",len(ts_StrValues))

        for j in range(m):
            timeSeries[i, j] = float(ts_StrValues[j + 1])
    file.close()
    return timeSeries,m

def StrToTS(line):
    tsStr = line.split(",")
    ts = np.empty(len(tsStr), dtype=float)
    for j in range(len(tsStr)):
        ts[j] = np.float(tsStr[j])
    return ts

def queryFileToTS(path):
    timeSeries=list()
    file=open(path,"r")
    line=file.readline()
    while line!='':
        #print(line)
        ts=StrToTS(line)
        timeSeries.append(ts)
        line=file.readline()
    return timeSeries



def accuracyPC(id_GT,id_App_KNN,K_NN):
    total=0
    for id in id_App_KNN:
        if id in id_GT:
            total+=1

    return total/K_NN



def App_KNN_Search(NN_values,query):
    newList=list(NN_values)
    newList.sort(key= lambda x:eucDistance(query,x[1]))

    id_App_KNN=list()
    for e in newList:
        id_App_KNN.append(e[0])
    return id_App_KNN


def GT_KNN_Search(timeSeries,query,nb_NN):
    NN = dict()
    id=0
    for ts in timeSeries:
        dist = eucDistance(query,ts)
        """
        if dist in NN.keys():
            raise Exception("double")
        """
        if len(NN)<nb_NN:
            NN[dist]=(id,ts)
        else:
            maxDist = max(NN)
            if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)
        id+=1

    newList = list(NN.values())

    newList.sort(key=lambda x: eucDistance(query, x[1]))

    id_GT_KNN = list()
    for e in newList:
        id_GT_KNN.append(e[0])
    return id_GT_KNN



def tsvToTxt(filePathTrain,filePathTest,file_name):
    lines_seen = set()  # holds lines already seen
    outfile = open("Data/"+file_name+".txt", "w")
    tsv_file = open(filePathTrain)
    n=0
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    for row in read_tsv:
        line = str(row[0])
        for i in range(1, len(row)):
            line += "," + str(row[i])
        if line not in lines_seen:  # not a duplicate
            outfile.write(line + "\n")
            lines_seen.add(line)
            n+=1
    tsv_file.close()
    tsv_file = open(filePathTest)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    m = None

    for row in read_tsv:
        m = len(row) - 1
        line = str(row[0])
        for i in range(1, len(row)):
            line += "," + str(row[i])
        if line not in lines_seen:  # not a duplicate
            outfile.write(line + "\n")
            lines_seen.add(line)
            n += 1
    tsv_file.close()

    m = m - (m % 10)
    outfile.write(str(n)+" "+str(m))
    outfile.close()
    return n,m
