import Util
import PASAX



def PASAX_GSSE_Precision(file_name):
    #print(file_name)

    timeSeries,m=Util.readDataset(file_name)


    nb_segment=m//10
    segments_size_start = 2
    

    indexes=PASAX.PASAX_GSSE(timeSeries, nb_segment, segments_size_start)
    segs_len=Util.segments_len(indexes)
    #print(segs_len)
    queriesPath="Queries/queries "+file_name
    queries=Util.queryFileToTS(queriesPath)
    nb_queries=len(queries)
    #print(nb_queries)


    nb_NN=10

    GT=list()
    app_KNN_SAX=list()
    app_KNN_SSE=list()



    ###############################################################################################################################
    for query in queries:

        GT.append(Util.GT_KNN_Search(timeSeries,query,nb_NN))

    ##############################################################################################################################
    for query in queries:
        query_PAA=Util.PAA_fixedSegSize(query,nb_segment)
        NN = dict()
        id=0
        for ts in timeSeries:
            ts_PAA = Util.PAA_fixedSegSize(ts,nb_segment)
            dist = Util.DR(query_PAA,ts_PAA,m)
            if len(NN)<nb_NN:
                NN[dist]=(id,ts)
            else:
                maxDist = max(NN)
                if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)

            id+=1

        app_KNN_SAX.append(Util.App_KNN_Search(NN.values(),query))

    #################################################################################################################################"


    for query in queries:
        query_PAA=Util.PAA_varSegSize(query,indexes)
        NN = dict()
        id=0
        for ts in timeSeries:
            ts_PAA = Util.PAA_varSegSize(ts,indexes)
            dist = Util.DR_VAR(query_PAA,ts_PAA,segs_len)
            if len(NN)<nb_NN:
                NN[dist]=(id,ts)
            else:
                maxDist = max(NN)
                if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)

            id+=1


        app_KNN_SSE.append(Util.App_KNN_Search(NN.values(),query))


    K_NN=10

    s=0
    for i in range(nb_queries):
        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_SAX[i][:K_NN],K_NN)
        s += p

    mean_SAX = s / nb_queries
    #print(mean_SAX)

    s=0
    for i in range(nb_queries):

        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_SSE[i][:K_NN],K_NN)
        s += p

    mean_SSE = s / nb_queries

    return mean_SAX,mean_SSE


##############################################################################################################################

def PASAX_LSSE_Precision(file_name):
    #print(file_name)

    timeSeries,m=Util.readDataset(file_name)


    nb_segment=m//10
    segments_size_start = 2
    

    indexes=PASAX.PASAX_LSSE(timeSeries, nb_segment, segments_size_start)
    segs_len=Util.segments_len(indexes)
    #print(segs_len)
    queriesPath="Queries/queries "+file_name
    queries=Util.queryFileToTS(queriesPath)
    nb_queries=len(queries)
    #print(nb_queries)


    nb_NN=10

    GT=list()
    app_KNN_SAX=list()
    app_KNN_SSE=list()



    ###############################################################################################################################
    for query in queries:

        GT.append(Util.GT_KNN_Search(timeSeries,query,nb_NN))

    ##############################################################################################################################
    for query in queries:
        query_PAA=Util.PAA_fixedSegSize(query,nb_segment)
        NN = dict()
        id=0
        for ts in timeSeries:
            ts_PAA = Util.PAA_fixedSegSize(ts,nb_segment)
            dist = Util.DR(query_PAA,ts_PAA,m)
            if len(NN)<nb_NN:
                NN[dist]=(id,ts)
            else:
                maxDist = max(NN)
                if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)

            id+=1

        app_KNN_SAX.append(Util.App_KNN_Search(NN.values(),query))

    #################################################################################################################################"


    for query in queries:
        query_PAA=Util.PAA_varSegSize(query,indexes)
        NN = dict()
        id=0
        for ts in timeSeries:
            ts_PAA = Util.PAA_varSegSize(ts,indexes)
            dist = Util.DR_VAR(query_PAA,ts_PAA,segs_len)
            if len(NN)<nb_NN:
                NN[dist]=(id,ts)
            else:
                maxDist = max(NN)
                if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)

            id+=1


        app_KNN_SSE.append(Util.App_KNN_Search(NN.values(),query))


    K_NN=10

    s=0
    for i in range(nb_queries):
        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_SAX[i][:K_NN],K_NN)
        s += p

    mean_SAX = s / nb_queries
    #print(mean_SAX)

    s=0
    for i in range(nb_queries):

        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_SSE[i][:K_NN],K_NN)
        s += p

    mean_SSE = s / nb_queries

    return mean_SAX,mean_SSE


##############################################################################################################################
