from copy import deepcopy
import numpy as np
from displayutils import *
from numpy import matlib

def aggregateScores_maxstd_dens_weight_predictor(data, attributes):   
    all_data =deepcopy(data)
   
    std = []        
    all_data = all_data.replace(-1, 0)
    for f in all_data.columns:
        std.append(all_data[f].std())
    
    features_std = set(zip(all_data.columns, std))
    features_max_std = []
    features_max = []
    for attr in attributes:
        max_std = 0
        max_feat = ''
        for f_v in features_std:
            if f_v[0].startswith(attr):
                if f_v[1]>max_std: 
                    max_std = f_v[1]
                    max_feat = f_v[0]
        if max_feat != '':
            features_max_std.append(max_std)
            features_max.append(max_feat)
    
    all_data = data[features_max]
       
    all_data = all_data.replace(-1.0,np.nan)
    #calculate the density based weights 
    feature_weights = []
    for c in all_data:
        nan_values = all_data[c].isna().sum()
        ratio = float(nan_values)/float(len(all_data[c]))
        feature_weights.append(1.0-ratio)

    weighted_columns = all_data*feature_weights
    weighted_columns_sum = weighted_columns.sum(axis=1, skipna=True)
    weighted_columns_mean = weighted_columns_sum/len(weighted_columns.columns)
    #rescale 
    weighted_columns_mean = np.interp(weighted_columns_mean, (weighted_columns_mean.min(), weighted_columns_mean.max()), (0, +1))
   
    return weighted_columns_mean

def aggregateScores_stdpredictor(data):   
    all_data =deepcopy(data)
   
    weights = []        
    all_data = all_data.replace(-1, 0)
    for f in all_data.columns:
        weights.append(all_data[f].std())
    
    #aggr_score = weighted_columns.sum(axis=1, skipna=True)/len(all_data.columns)
    weighted_columns = all_data*weights

    aggr_score = weighted_columns.apply(lambda row: row.sum()/row.count(), axis=1)
    aggr_rescaled = np.interp(aggr_score.values, (aggr_score.min(), aggr_score.max()), (0, +1))
    return aggr_rescaled
        
def aggregateScores(data):
    #cosine tfidf if exists receives a weight of 0.5
    #all other columns share 0.5 and are additionally weighted by their density
    if ('cosine_tfidf' in data.columns.values):
        cosine_tfidf_column = data['cosine_tfidf']
        other_columns  = data.drop(['cosine_tfidf'], axis=1)
        other_columns = other_columns.replace(-1.0,np.nan)
    
    else: 
        other_columns=data
        other_columns = other_columns.replace(-1.0,np.nan)
    #calculate the density based weights for the other columns
    column_weights = []
    for c in other_columns:
        nan_values = other_columns[c].isna().sum()
        ratio = float(nan_values)/float(len(other_columns[c]))
        column_weights.append(1.0-ratio)

    weighted_columns = other_columns*column_weights
    other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
    other_columns_mean = other_columns_sum/len(other_columns.columns)
    #rescale 
    other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))
    if ('cosine_tfidf' in data.columns.values):
        cosine_tfidf_column = np.interp(cosine_tfidf_column, (cosine_tfidf_column.min(), cosine_tfidf_column.max()), (0, +1))
        weighted_cosine = cosine_tfidf_column*0.5
        weighted_other_columns = other_columns_mean*0.5   
        sum_weighted_similarity = weighted_other_columns+weighted_cosine
        return sum_weighted_similarity
    
    else: return other_columns_mean

def calculateThreshold(scores, threshold_type):
    if threshold_type == 'otsu':
        return otsus_threshold(scores)
    elif threshold_type == 'valley':
        return valley_threshold(scores)   
    elif threshold_type == 'static':
        return 0.5;
    elif threshold_type == 'elbow':
        return elbow_threshold(scores)
    else: 
        print("Unknown threshold method. Default is static, t=0.5")
        return 0.5;
        
 
#code from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
def elbow_threshold(scores):
    similarities = deepcopy(scores)
    similarities.sort()
    sim_list = list(similarities)
    nPoints = len(sim_list)
    allCoord = np.vstack((range(nPoints), sim_list)).T
    
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel    
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
  
    #print ("Elbow Threshold: "+str(similarities[idxOfBestPoint]))
       
    return similarities[idxOfBestPoint]
    

def otsus_threshold(scores):
    similarities = deepcopy(scores)
    
    similarities[::-1].sort() #sort descending
    hist, _ = np.histogram(similarities, bins=len(similarities), range=(0.0, 1.0))
    hist = 1.0 * hist / np.sum(hist)
    val_max = -999
    thr = -1
    print_progress(1, len(similarities) - 1, prefix="Find Otsu's threshold:", suffix='Complete')
    for t in range(1, len(similarities) - 1):
        print_progress(t + 1, len(similarities) - 1, prefix="Find Otsu's threshold:", suffix='Complete')
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        if q1 != 0 and q2 != 0:
            m1 = np.sum(np.array([ i for i in range(t) ]) * hist[:t]) / q1
            m2 = np.sum(np.array([ i for i in range(t, len(similarities)) ]) * hist[t:]) / q2
            val = q1 * (1 - q1) * np.power(m1 - m2, 2)
            if val_max < val:
                val_max = val
                thr = similarities[t]

    print ("Otsu's Threshold: %f " % thr)
    return thr

def valley_threshold(scores):
    similarities = deepcopy(scores)
    similarities[::-1].sort() #sort descending
    hist, _ = np.histogram(similarities, bins=len(similarities), range=(0.0, 1.0))
    hist = 1.0 * hist / np.sum(hist)
    val_max = -999
    thr = -1
    print_progress(1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
    float_list = [round(elem, 2) for elem in similarities]
    #normalizes by occurrences of most frequent value
    fre_occur = float_list.count(max(float_list,key=float_list.count))
    for t in range(1, len(similarities) - 1):
        print_progress(t + 1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        if q1 != 0 and q2 != 0:
            m1 = np.sum(np.array([ i for i in range(t) ]) * hist[:t]) / q1
            m2 = np.sum(np.array([ i for i in range(t, len(similarities)) ]) * hist[t:]) / q2

            val = (1.0-float(float_list.count(round(similarities[t],2)))/float(fre_occur))*(q1 * (1.0 - q1) * np.power(m1 - m2, 2))
            if val_max < val:
                val_max = val
                thr = similarities[t]
    
    
    print ("Valley Threshold: %f " % thr)
    return thr
    
def getUnsupervisedResults(data):

    true_positives = data[(data.label) & (data.unsupervised_label)]
    true_negatives = data[(data.label==False) & (data.unsupervised_label==False)]
    false_negatives = data[(data.label==True) & (data.unsupervised_label==False)]
    false_positives = data[(data.label==False) & (data.unsupervised_label==True)]

    precision = len(true_positives)/(len(true_positives)+len(false_positives))
    recall = len(true_positives)/(len(true_positives)+len(false_negatives))
    f1 = 2*precision*recall/(precision+recall)

    print("Precision: %f ---- Recall: %f ---- F1: %f" %(precision,recall,f1))
    return (precision,recall,f1)