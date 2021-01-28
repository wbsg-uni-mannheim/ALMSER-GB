from __future__ import division
from similarity.levenshtein import Levenshtein
from similarity.jaccard import Jaccard
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
import re
from datetime import tzinfo
from dateutil.parser import parse
import pytz
from numpy import trapz
import re
from scipy.ndimage import gaussian_filter
from numpy import matlib
from copy import deepcopy
import nltk
from nltk import ngrams
import matplotlib.pyplot as plt
from datautils import *

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


def get_num_equal(num1, num2):
    if num1 == 'nan' or num2 == 'nan' or num1 == '' or num2 == '':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        if num1_ == num2_:
            return 1.0
        return 0.0
    except:
        return -1

def get_norm_sim(num1,num2,max_value, min_value):
    if num1 == 'nan' or num2 == 'nan' or  num1 == '' or num2 == '':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        abs_difference = abs(num1_ - num2_)
        return 1-(abs(abs_difference - min_value)/(max_value-min_value))
    except:
        return -1

def get_abs_diff(num1, num2):
    if num1 == 'nan' or num2 == 'nan' or  num1 == '' or num2 == '':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        return abs(num1_ - num2_)
    except:
        return -1

def get_jaccard_token_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        return 1-nltk.jaccard_distance(set(str1), set(str2))
    
    
def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        return float(len(c)) / float(len(a) + len(b) - len(c))


def get_relaxed_jaccard_sim(str1, str2, n_grams=1):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '' :
        return -1.0
    a = set(str1.split())
    b = set(str2.split())
    if not a or not b: return -1
    c = []
    for a_ in a:
        for b_ in b:
            if get_levenshtein_sim(a_, b_) > 0.7:
                c.append(a_)
    intersection = len(c)
    min_length = min(len(a), len(b))
    if intersection > min_length:
        intersection = min_length
    return float(intersection) / float(len(a) + len(b) - intersection)


def get_containment_sim(str1, str2, allowTokenBased= True):
    #it's not really a long string necessarily but it does not make sense to do word based containment
    if ((len(set(str1.split()))>1 and len(set(str2.split()))>1) or not allowTokenBased): 
        a = set(str1.split())
        b = set(str2.split())
    else: #for single words we consider the tokens
        if (allowTokenBased):
            a = set(str1)
            b = set(str2)
       
        
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif len(a) == 0 or len(b) == 0:
        return -1.0
    else:
        return float(len(c)) / float(min(len(a), len(b)))


def get_levenshtein_sim(str1, str2):
    levenshtein = Levenshtein()
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        max_length = max(len(str1), len(str2))
        return 1.0 - levenshtein.distance(str1, str2) / max_length


def get_missing(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '' :
        return 1.0
    else:
        return 0.0


def get_overlap_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif str1 == str2:
        return 1.0
    else:
        return 0.0


def get_cosine_word2vec(str1, str2, model):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif str1.replace(' ', '') in model.vocab and str2.replace(' ', '') in model.vocab:
        return model.similarity(str1.replace(' ', ''), str2.replace(' ', ''))
    else:
        return 0.0


def get_cosine_tfidf(tfidf_scores_ids, sourceID, targetID):
    try:
        source_index = np.where(tfidf_scores_ids['ids'] == sourceID)
        target_index = np.where(tfidf_scores_ids['ids'] == targetID)
        score = cosine_similarity(tfidf_scores_ids['scores'][source_index].todense(), tfidf_scores_ids['scores'][target_index].todense())
    except:
        import pdb; pdb.set_trace();
    return score[0][0]
    

def calculateTFIDF(records, grams=1): 
    try:
        records_data = records['data']
        concat_records = []
        for row in records_data:
            if (isinstance(row,np.ndarray)): # tfidf based on  more that one features
                concat_row = ''
                for value in row:
                    if not pd.isnull(value):
                        if type(value) is str:
                            if value.lower() != 'nan':
                                value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value)) #think of product model names e.g. ak-123
                                concat_row += ' ' + value
                        else: # tfidf based on one feature 
                            value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                            concat_row += ' ' + str(value)

                concat_records.append(concat_row.lower())
            else: 
                if pd.isnull(row):
                    concat_records.append("")
                else:
                    value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row))
                    concat_records.append(value.lower())

        tf_idfscores = TfidfVectorizer(encoding='latin-1', ngram_range=(grams,grams)).fit_transform(concat_records)
        tf_idf = dict()
        tf_idf['ids'] = records['ids']
        tf_idf['scores'] = tf_idfscores
    except Exception as e:
        print(str(e))
        import pdb;pdb.set_trace();
    return tf_idf
