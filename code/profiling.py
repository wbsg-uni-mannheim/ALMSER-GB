import datetime
import copy
import re
import numpy as np
from sklearn.tree import _tree
import matplotlib.pyplot as plt
import statistics


def getFeatureDensities(feature_vector, common_attributes, show=False):
    feature_vector_with_nulls = copy.copy(feature_vector)
    feature_vector_with_nulls = feature_vector_with_nulls.replace({-1: None})
    non_null_values = feature_vector_with_nulls.count()
    density = round(feature_vector_with_nulls.count()/len(feature_vector_with_nulls.index),2)
    visited = []
    overall_density = []
    if show:
        print("*****Feature densities*****")
    for feat in feature_vector_with_nulls.columns:
        if (feat not in ['source_id', 'target_id', 'pair_id', 'label']):
            for common_attr in common_attributes:
                if (feat.startswith(common_attr) and common_attr not in visited):
                    visited.append(common_attr)
                    overall_density.append(density[feat])
                    if show: print(common_attr+": "+str(density[feat]))
    return statistics.mean(overall_density)

def getUniqueness(feature_vector, column):
    column_values= copy.copy(feature_vector[column])
    distinct_values = column_values.nunique()
    non_null_values = column_values.count()
    if non_null_values==0: uniqueness=0
    else : uniqueness = round(float(distinct_values)/non_null_values,2)
    return uniqueness

def getAvgStdLength(feature_vector, column, mode):
    column_values= copy.copy(feature_vector[column])
    column_values.fillna('nan', inplace=True)
    lengths = []       
    for i in column_values.values:
        if i!='nan': 
            if mode == 'tokens' : lengths.append(len(str(i)) )
            elif mode == 'words' : lengths.append(len(str(i).split()))
    avg = 0 if len(lengths) == 0 else round(np.mean(lengths),2)
    std = 0 if len(lengths) == 0 else round(np.std(lengths),2)
    
    return avg,std

def getDensity(feature_vector, column):
    density = round(feature_vector[column].count()/len(feature_vector.index),3)
    return density

#data
def getTypesofData(feature_vector, column):
    moretypes=False # if more than one types are detected
    
    column_values = feature_vector[column].dropna()

    type_list=list(set(column_values.map(type).tolist()))

    if len(type_list) == 0: 
        "No type could be detected. Default (string) will be assigned."
        datatype = 'str'
    elif len(type_list) >1: 
        "More than one types could be detected. Default (string) will be assigned."
        datatype = 'str'
    else:            
        if str in type_list:   
            types_of_column = []
            length = 0 
            for value in column_values:
                length = length + len(value.split())
                if re.match(r'.?\d{2,4}[-\.\\]\d{2}[-\.\\]\d{2,4}.?', value):
                    #check if it can be really converted
                    date_value = get_date_type(value)
                    if date_value != None:types_of_column.append('date')

            avg_length = length/len(column_values)

            if (avg_length>6): types_of_column.append('long_str')
            if len(set(types_of_column)) <= 1:                  
                if ('date' in types_of_column and (types_of_column.count('date')> (len(column_values)/2))):
                    # assign date if you found the date type for the majority of thev alues
                    datatype = 'date'
                elif ('long_str' in types_of_column):
                    datatype = 'long_str'
                else : datatype = 'str'
            else: 
                print("More than one types could be detected. Default (string) will be assigned.")
                datatype = 'str'
                moretypes= True
        else: # else it must be numeric
            datatype = 'numeric'
    return datatype, moretypes

def get_date_type(date_str):
    separator = ''
    if '.' in date_str:
        separator = '.'
    elif '\\' in date_str:
        separator = '\\'
    elif '/' in date_str:
        separator = '/'
    elif '-' in date_str:
        separator = '-'
    else:
        return None
    try:
        date_parts = [ d.strip() for d in date_str.split(separator) ]
        if re.match('\\d{4}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            return datetime.datetime.strptime(date_str, '%Y' + separator + '%m' + separator + '%d').date()
        if re.match('\\d{1,2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{4}', date_str):
            return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%Y').date()
        if re.match('\\d{2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            p = re.compile('\\d+')
            splitted_date = p.findall(date_str)
            if int(splitted_date[0]) < 32 and int(splitted_date[1]) < 13:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            if int(splitted_date[0]) > 32:
                return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
            try:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            except:
                try:
                    return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
                except:
                    display('Unknown pattern or invalid date: %s' % date_str)
                    return None

        else:
            return parse(date_str, fuzzy=True)
    except:
        f = open('unparseddates.txt', 'a')
        f.write(date_str + '\n')
        f.close()
        return None

def printTreeRules(feature_names, tree):
    tree_model = []
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #tree_model =  tree_model +"def tree({}):".format(", ".join(feature_names))
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            tree_model.append("{}if {} <= {}:".format(indent, name, round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            tree_model.append("{}else:  # if {} > {}".format(indent, name, round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            tree_model.append("{}return {}".format(indent, tree_.value[node]))
    
    recurse(0, 1)
    return ''.join(tree_model)

def get_model_importances(model,classifierName=None):
 
    if classifierName == 'logr':
        importances = model.coef_.ravel()
    elif classifierName == 'svm':
        if model.kernel != 'linear':
            display("Cannot print feature importances without a linear kernel")
            return
        else: importances = model.coef_.ravel()
    else:
        importances = model.feature_importances_
    
    return importances

def getAvgLength(feature_vector, column, mode):
    column_values= copy.copy(feature_vector[column])
    column_values.fillna('nan', inplace=True)
    lengths = []       
    for i in column_values.values:
        if i!='nan': 
            if mode == 'tokens' : lengths.append(len(str(i)) )
            elif mode == 'words' : lengths.append(len(str(i).split()))

    avg = 0 if len(lengths) == 0 else round(float(sum(lengths) / len(lengths)),2)
    return avg

def showFeatureImportances(column_names, model, classifierName,display=True):
      
    importances = get_model_importances(model, classifierName)
       
    column_names = [c.replace('<http://schema.org/Product/', '').replace('>','') for c in column_names]
    sorted_zipped = sorted(list(zip(column_names, importances)), key = lambda x: x[1], reverse=True)[:50]
   
    features_in_order = [val[0] for val in sorted_zipped]
    feature_weights_in_order = [round(val[1],2) for val in sorted_zipped]
    if (display):
        plt.figure(figsize=(18,3))
        plt.title('Feature importances for classifier %s (max. top 50 features)' % classifierName)
        plt.bar(range(len(sorted_zipped)), [val[1] for val in sorted_zipped], align='center', width = 0.8)
        plt.xticks(range(len(sorted_zipped)), [val[0] for val in sorted_zipped])
        plt.xticks(rotation=90)
        plt.show() 

    return features_in_order,feature_weights_in_order

def getCornerCases(feature_vector, attribute, getHardRatios=False):
    positives = feature_vector[feature_vector['label']==True]
    negatives = feature_vector[feature_vector['label']==False]

    bins = [-1.0,0,0.2,0.4,0.6,0.8,1.0]
    bin_positives = pd.cut(positives[attribute], bins=bins).value_counts(sort=False)
    bin_negatives = pd.cut(negatives[attribute], bins=bins).value_counts(sort=False)
    ratio_hard_neg = float(len(negatives[negatives[attribute]>0.8].index))/float(len(feature_vector.index))
    ratio_hard_pos = float(len(positives[positives[attribute]<0.2].index))/float(len(feature_vector.index))
    
    if (getHardRatios): return bin_positives, bin_negatives,ratio_hard_neg,ratio_hard_pos
    else: return bin_positives, bin_negatives
    
def get_cor_attribute(common_attributes, pairwiseatt):
    for c_att in common_attributes:
        if  pairwiseatt.startswith(c_att): return c_att
        if c_att.startswith('cosine_tfidf'): return 'all'
        
def getCornerCaseswithOptimalThreshold(feature_vector, attributes):
   
    positives = copy.copy(feature_vector[feature_vector['label']==True])
    negatives = copy.copy(feature_vector[feature_vector['label']==False])
    
    positives = positives.replace(-1, 0)
    negatives = negatives.replace(-1, 0)
    
    positive_values = positives[attributes].mean(axis=1).values
    negative_values = negatives[attributes].mean(axis=1).values
    
    thresholds = []
    fp_fn = []
    for t in np.arange(0.0, 1.01, 0.01):
        fn = len(np.where(positive_values<t)[0])
        fp = len(np.where(negative_values>=t)[0])
        thresholds.append(t)
        fp_fn.append(fn+fp)
    

    optimal_threshold = thresholds[fp_fn.index(min(fp_fn))]
    hard_cases = min(fp_fn)
    groups_positives = positives[attributes].groupby(attributes).size().reset_index()
    return hard_cases/len(positive_values),groups_positives.shape[0]/len(positive_values)