from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score, cross_validate,cross_val_predict, StratifiedShuffleSplit, GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
from profiling import *
import pandas as pd
from sklearn.tree import export_text
import seaborn as sns;
from tqdm import tqdm


def getClassifier(name, *args, **kwargs):
    if name=="rf": return RandomForestClassifier(*args, **kwargs)
    elif name=="gboost" : return GradientBoostingClassifier(*args, **kwargs)
    elif name=="svm" : return SVC(*args, **kwargs)
    elif name=="dt": return DecisionTreeClassifier(*args, **kwargs)
    elif name=="logr": return LogisticRegression(*args, **kwargs)
    elif name=="linr": return LinearRegression(*args, **kwargs)
    else: 
        print("Unknown classifier name %s" %s)
        return None

    
def getFeatureImportances(model):
    if isinstance(model, SVC) and model.kernel != 'linear':
        display("Cannot print feature importances without a linear kernel")
        return None
    elif isinstance(model, SVC) or isinstance(model, LogisticRegression) or isinstance(model, LinearRegression):
        return model.coef_.ravel()
    else: return model.feature_importances_

def getNestedXValidationResults(data, model):
    
    if ('source_id' in data.columns.values):
        X = data.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
    else: X = data.drop(['label'], axis=1)
    y = data['label'].values
    f1_scorer = make_scorer(f1_score)

    if model=="non-linear":
        grid_values = {'n_estimators' : [10, 100, 500], 'max_depth' : [10, 50, None], 'min_samples_leaf' : [1, 3, 5]}
        clf = RandomForestClassifier(random_state=1)

    elif model=="linear":
        grid_values_1 = { 'kernel' : ['linear'], 'max_iter':[100000]}
        grid_values_2 = {'C' : np.logspace(-2, 5, 10), 'max_iter':[100000], 'kernel' : ['rbf'], 'gamma': [1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+01, 'scale']}
        grid_values= []
        grid_values.append(grid_values_1)
        grid_values.append(grid_values_2)
        clf = SVC(random_state=1, probability=True)

    clf_gs = GridSearchCV(clf, grid_values, scoring=f1_scorer, cv=StratifiedShuffleSplit(n_splits=4,random_state =1), verbose=10, n_jobs=20)
    xval_scoring = {'precision' : make_scorer(precision_score),
                           'recall' : make_scorer(recall_score), 'f1_score' : make_scorer(f1_score)}

    results = cross_validate(clf_gs, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring)
    prediction_probs = cross_val_predict(clf_gs, X, y, method='predict_proba')
    pred_scores = list(map(lambda x:max(x),prediction_probs))

    precision = round(np.mean(results['test_precision']),2)
    recall = round(np.mean(results['test_recall']),2)
    f1 = round(np.mean(results['test_f1_score']),2)
    f1_std = round(np.std(results['test_f1_score']),2)
    
    print('Precision : '+str(precision))
    print('Recall : '+str(recall))
    print('F1 : '+str(f1))
    print('F1_std : '+str(f1_std))
    
    #print("--- Feature Importances ---")
    #showFeatureImportances(X.columns.values, clf, classifierName='tree')
    return precision, recall, f1, f1_std

def getSplitValidationResults(train_val, test, model_type, optimization= False, model_name = None, *args, **kwargs):
   
    X_train_val = train_val.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
    y_train_val =train_val['label'].values
    
    X_test = test.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
    y_test = test['label'].values

    f1_scorer = make_scorer(f1_score)
    
    # source:https://www.wellformedness.com/blog/using-a-fixed-training-development-test-split-in-sklearn/
    if optimization:
        splits_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
        train_index, val_index = next(splits_train_val.split(X_train_val,y_train_val))

        X_train, X_val, y_train, y_val = X_train_val.loc[train_index], X_train_val.loc[val_index],y_train_val[train_index],y_train_val[val_index]
    
        X_train_val = np.concatenate([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        val_fold = np.concatenate([
            np.full(X_train.shape[0], -1 ,dtype=np.int8),
            # The development data.
            np.zeros(X_val.shape[0], dtype=np.int8)
        ])
        cv = PredefinedSplit(val_fold)

        if model_type=="non-linear":
            grid_values = {'n_estimators' : [10, 100, 500], 'max_depth' : [5, 10, 50, None], 'min_samples_leaf' : [1, 3, 5]}
            clf = RandomForestClassifier(random_state=1)

        elif model_type=="linear":
            grid_values_1 = { 'kernel' : ['linear'], 'max_iter':[100000]}
            grid_values_2 = {'C' : np.logspace(-2, 5, 10), 'max_iter':[100000], 'kernel' : ['rbf'], 'gamma': [1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+01, 'scale']}
            grid_values= []
            grid_values.append(grid_values_1)
            grid_values.append(grid_values_2)
            clf = SVC(random_state=1, probability=True)
    
        model = GridSearchCV(clf, grid_values, scoring=f1_scorer, cv=cv,verbose=10, n_jobs=-1) #parallel jobs
    
    else: 
        if model_type=="non-linear":
            if model_name=="rf" or model_name==None:
                model = RandomForestClassifier(random_state=1, *args, **kwargs)
            if model_name=="dt":
                model = DecisionTreeClassifier(random_state=1, *args, **kwargs)
        elif model_type == "linear":
            model = SVC(random_state=1, probability=True, *args, **kwargs)
    
    model.fit(X_train_val,y_train_val)
    if(model_name=="dt"):
        #print(printTreeRules( X_train_val.columns,model))
        text_rules = export_text(model, feature_names=X_train_val.columns.values.tolist())
        print(text_rules)
    
    if (bool(X_test.columns.values.tolist()!=X_train_val.columns.values.tolist()) ):
        print("Columns of train and test set not aligned. STOP.")
    predictions = model.predict(X_test)
    prediction_probs = model.predict_proba(X_test)
    pred_scores = list(map(lambda x:max(x),prediction_probs))
    prec, recall, fscore, support  = precision_recall_fscore_support(y_test, predictions, average='binary')

    precision =round(prec,2)
    recall =round(recall,2)
    f1 =round(fscore,2)

    return precision, recall, f1

def getHeatmapOfSetting(fv, distinct_sources):
    
    print("Calculate and display heatmap of the complete matching setting")
    results_hmap = pd.DataFrame(index=sorted(list(distinct_sources)), columns=sorted(list(distinct_sources)))
    results = pd.DataFrame(columns= ['source_pair', 'target_pair','precision','recall','f1'])
    
    for source in tqdm(sorted(list(distinct_sources))):
        for target in sorted(list(distinct_sources)): 
            #print ("Source - Target: %s - %s " %(source,target))
            fv_source = fv[fv.datasource_pair==source]
            fv_target = fv[fv.datasource_pair==target]

            metadata_columns = ['source_id','target_id','pair_id','datasource_pair', 'agg_score', 'unsupervised_label', 'source', 'target']
            selected_columns = [ele for ele in fv_source.columns.values.tolist() if ele not in metadata_columns]
               
            feature_vector_source = fv_source[selected_columns]
            feature_vector_target = fv_target[selected_columns]
            #the order of columns matters!!!!
            feature_vector_target = feature_vector_target[feature_vector_source.columns]

            if (source is target):      
                X = feature_vector_source.drop(['label'], axis=1)
                y = feature_vector_source['label'].values

                clf = RandomForestClassifier(random_state=1)
                xval_scoring = {'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score),
                                'f1_score' : make_scorer(f1_score)}            
                max_result = cross_validate(clf, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring, n_jobs=-1)
                f1 = round(np.mean(max_result['test_f1_score']),2)
                precision = round(np.mean(max_result['test_precision']),2)
                recall = round(np.mean(max_result['test_recall']),2)

                #print("F1 for %s is %f" %(source,f1))

            else:
                X_train = feature_vector_source.drop(['label'], axis=1)
                y_train = feature_vector_source['label'].values
                
                X_test = feature_vector_target.drop(['label'], axis=1)
                y_test = feature_vector_target['label'].values
                
                model = RandomForestClassifier(random_state=1)
                model.fit(X_train,y_train)
                predictions = model.predict(X_test)
                prec, recall, f1, support  = precision_recall_fscore_support(y_test, predictions, average='binary')
            
            results_hmap.loc[source, target] = f1
            results.loc[results.shape[0]] = [source, target, precision, recall, f1] 
    
    heatm = results_hmap.fillna(value=1)
    ax = sns.heatmap(heatm, annot=True)
    plt.show()
    all_scores = results_hmap.values.flatten()
    ntl_scores = []
    rem_index = []
    for i in range(0, len(all_scores), 11):
        rem_index.append(i)

    for s in range(0,len(all_scores)):
        if s not in rem_index: ntl_scores.append(all_scores[s])

    print("Mean of Naive Transfer Learning Results: ", statistics.mean(ntl_scores))
    print("Median of Naive Transfer Learning Results: ", statistics.median(ntl_scores))