from graphutils import *
from networkx.algorithms.components import *
from networkx.algorithms.centrality import *
import pdb
from learningutils import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from displayutils import *
import time 
import random
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import statistics
from scipy.stats import entropy
from ALMSER_utils import *
from sklearn.model_selection import train_test_split
import itertools
from ALMSER_log import *


class ALMSER(object):
    
    random.seed(15)
    
    def __init__(self, feature_vector_train, feature_vector_test, unique_sources, quota, classifier_name, query_strategy, fvsplitter, rltd,  bootstrap=True, use_origin=False, groups=None, rltd_top_setting=None):        
        
        self.criteria = get_criteria_list(query_strategy)
        
        
        unlabeled_set_ids = list(feature_vector_train['source']+"-"+feature_vector_train['target'])
        unlabeled_set_ds_pair = list(map(lambda x: x.split("-")[0].rsplit("_",1)[0]+fvsplitter+x.split("-")[1].rsplit("_",1)[0], unlabeled_set_ids))
       
        self.labeled_set =  labeled_set_init()       
        self.unlabeled_set = unlabeled_set_init()
        
        self.unlabeled_set['source'] = feature_vector_train['source']
        self.unlabeled_set['target'] = feature_vector_train['target']
        self.unlabeled_set['unsupervised_label'] = feature_vector_train['unsupervised_label']
        self.unlabeled_set['datasource_pair'] = unlabeled_set_ds_pair
        
        metadata_columns = ['source_id','target_id','pair_id','agg_score', 'unsupervised_label','datasource_pair']
        
        features_columns = [ele for ele in feature_vector_train.columns.values.tolist() if ele not in metadata_columns]
        self.feature_vector = feature_vector_train[features_columns]
        
        metadata_columns_in_pool_and_gs = metadata_columns+['source','target', 'label']

        self.unlabeled_set_metadata = copy.copy(feature_vector_train[metadata_columns_in_pool_and_gs])
         
        
        self.gs = feature_vector_test[features_columns]
        self.gs_metadata = feature_vector_test[metadata_columns_in_pool_and_gs]
        self.quota = quota
        self.unique_source_pairs = unique_sources
        self.count_sources()
        self.classifier_name = classifier_name
        self.query_strategy = query_strategy
        self.rltd = rltd
        if rltd_top_setting is None:
            self.rltd_top_setting = 'unassigned'
        else: self.rltd_top_setting=rltd_top_setting

        #initialize the pair-source learning models and the overall learning model
        self.learning_models = dict()
        self.learning_models_all_minus_one = dict()

        self.results = results_init(self.quota)
        self.informants_eval = pd.DataFrame()
        self.G = nx.Graph()

        self.groups_of_tasks = groups
        if (groups):
            print("Groups of matching tasks to consider:", self.groups_of_tasks)
        
        #bootstrap AL model by training the model of iteration 0 with the unsupervised labels
        print("Bootstrap model")
        model = getClassifier(self.classifier_name, n_estimators=10, warm_start=True)
        self.learning_models['boot_all'] = model.fit(self.feature_vector.drop(['label','source','target'], axis=1), self.unlabeled_set['unsupervised_label'])
        
        #bootstrap the labeled set
        if (bootstrap):
            print("Bootstrap labeled set")
            self.bootstrap_labeled_set()
    
        self.tasks_to_exploit=[]
        self.phase= ""
        
        self.log = ALMSER_log(self.quota)

    def calculate_graph_info(self):

        self.unlabeled_set.graph_inferred_label = self.unlabeled_set.apply(lambda row, G=self.G: has_path_(G, row.source, row.target), axis=1)
       
       # prec, recall, fscore, support  = precision_recall_fscore_support(self.unlabeled_set_metadata.label, self.unlabeled_set.graph_inferred_label, average='binary')
      
    def calculate_uncertainty(self, unlabeled_data):
        
        labeled_X, labeled_y = self.get_feature_vector_subset(self.labeled_set)
        
        clf= getClassifier('svm').fit(labeled_X, labeled_y)
        uncertainty = np.abs(clf.decision_function(unlabeled_data)).tolist()
        return uncertainty
    
    def calculate_disagreement(self, unlabeled_data, graph_signal, correct_prediction=False):
        labeled_X, labeled_y = self.get_feature_vector_subset(self.labeled_set)

        m1_pred = self.unlabeled_set['predicted_label'].values
        
        m2 = getClassifier('dt').fit(labeled_X, labeled_y)
        m2_pred = m2.predict(unlabeled_data).tolist()

        m3 = getClassifier('gboost').fit(labeled_X, labeled_y)
        m3_pred = m3.predict(unlabeled_data).tolist()

        m4 = getClassifier('logr').fit(labeled_X, labeled_y)
        m4_pred = m4.predict(unlabeled_data).tolist()
      
        m5 = getClassifier('svm',probability=True).fit(labeled_X, labeled_y)
        m5_pred = m5.predict(unlabeled_data).tolist()
        
        
        if graph_signal:
            m1_pred = self.unlabeled_set['graph_inferred_label'].values

        all_predictions = list(zip(m1_pred, m2_pred, m3_pred, m4_pred, m5_pred))

        if correct_prediction:
            disagreement = list(map(lambda x: disagreement_score(x), all_predictions))
        else: 
            disagreement = list(map(lambda x: 1 - max(x.count(True),x.count(False))/len(x), all_predictions))
                
        self.calculateF1ofInformants(all_predictions)
        
        return disagreement,all_predictions
    
    
    def get_labeledset_current_status(self):
        source_pair_labeled_set = self.labeled_set['datasource_pair'].values.tolist()  
        status = dict()
        for sp in self.unique_source_pairs:
            status[sp] = round(source_pair_labeled_set.count(sp)/len(self.labeled_set),3)
        return status
    
    def get_labeledsetgroup_current_status(self):
   
        groups = []
        for sp in self.labeled_set['datasource_pair'].values.tolist():
            groups.append(self.groups_of_tasks.get(sp))
        
        #group frequency per data source pair
        status = dict()
        for sp in self.labeled_set['datasource_pair'].values.tolist():
            status[sp] = round(groups.count(self.groups_of_tasks.get(sp))/len(self.labeled_set),3)
        
        return status
    
    def get_unlabeledset_current_status(self):
        source_pair_unlabeled_set = self.unlabeled_set['datasource_pair'].values.tolist()  
        status = dict()
        for sp in self.unique_source_pairs:
            status[sp] = round(source_pair_unlabeled_set.count(sp)/len(self.unlabeled_set),3)
        return status
    

    def bootstrap_labeled_set(self):    
        idx_max_scores = self.unlabeled_set_metadata.groupby("datasource_pair")['agg_score'].transform(max) == self.unlabeled_set_metadata['agg_score']
        idx_min_scores = self.unlabeled_set_metadata.groupby("datasource_pair")['agg_score'].transform(min) == self.unlabeled_set_metadata['agg_score']

        #now select one random pair per datasource
        how_many_to_select = 1
        if (self.groups_of_tasks):
            how_many_to_select = int(len(self.groups_of_tasks.keys())/len(self.groups_of_tasks.values()))
        
        max_indices_boot = self.unlabeled_set_metadata[idx_max_scores].groupby('datasource_pair').apply(lambda x: x.sample(how_many_to_select,random_state=42).index)
        min_indices_boot = self.unlabeled_set_metadata[idx_min_scores].groupby('datasource_pair').apply(lambda x: x.sample(how_many_to_select,random_state=42).index)   
        

        for pos_boot_idx in max_indices_boot:
            boot_pair = self.unlabeled_set_metadata.loc[pos_boot_idx]
            if (not boot_pair.label.values): print("Bootstrapped as positive but it is actually negative")
            
            self.labeled_set = self.labeled_set.append({'source': boot_pair.source.values[0], 'target':boot_pair.target.values[0], 'label':True, 'datasource_pair':boot_pair.datasource_pair.values[0], 'predicted_label':'bootstrap'}, ignore_index=True) 

            self.unlabeled_set.drop(pos_boot_idx, inplace=True)
            self.unlabeled_set_metadata.drop(pos_boot_idx, inplace=True)
        

        for neg_boot_idx in min_indices_boot:
            boot_pair = self.unlabeled_set_metadata.loc[neg_boot_idx]
            if (boot_pair.label.values): 
                print("Bootstrapped as negative but it is actually positive")
            
            self.labeled_set = self.labeled_set.append({'source': boot_pair.source.values[0], 'target':boot_pair.target.values[0], 'label':False, 'datasource_pair':boot_pair.datasource_pair.values[0], 'predicted_label':'bootstrap'}, ignore_index=True) 

            self.unlabeled_set.drop(neg_boot_idx, inplace=True)
            self.unlabeled_set_metadata.drop(neg_boot_idx, inplace=True)

        self.update_learning_models_per_ds(boot_state=True)
       
            
    def update_after_answer(self, candidate_idx):
        r1 = self.unlabeled_set.loc[candidate_idx]['source']
        r2 = self.unlabeled_set.loc[candidate_idx]['target']

        true_label= self.unlabeled_set_metadata.loc[candidate_idx]['label']
        datasource_pair = self.unlabeled_set_metadata.loc[candidate_idx]['datasource_pair']
        
        disagreement= self.unlabeled_set.loc[candidate_idx]['disagreement']
        datasource_pair_frequency= self.unlabeled_set.loc[candidate_idx]['datasource_pair_frequency']


        inf_score= self.unlabeled_set.loc[candidate_idx]['inf_score']
        graph_inferred_label = self.unlabeled_set.loc[candidate_idx]['graph_inferred_label']
        predicted_label = self.unlabeled_set.loc[candidate_idx]['predicted_label']
        votes = self.unlabeled_set.loc[candidate_idx]['votes']
        sel_proba = self.unlabeled_set.loc[candidate_idx]['sel_proba']
        cc_size = self.unlabeled_set.loc[candidate_idx]['graph_cc_size']
        task_rltd = self.unlabeled_set.loc[candidate_idx]['task_rltd']


        self.labeled_set = self.labeled_set.append({'source': r1, 'target':r2, 'label':true_label, 'datasource_pair':datasource_pair,'disagreement':disagreement, 'datasource_pair_frequency':datasource_pair_frequency, 'inf_score':inf_score, 'graph_inferred_label': graph_inferred_label, 'sel_proba':sel_proba, 'cc_size':cc_size, 'predicted_label': predicted_label, 'votes':votes, 'task_rltd':task_rltd}, ignore_index=True)
               
        self.unlabeled_set.drop(candidate_idx, inplace=True)
        self.unlabeled_set_metadata.drop(candidate_idx, inplace=True)

        self.update_learning_models_per_ds()
       
        return r1,r2,true_label
    
    def update_learning_models_all_minus_one(self):
        #update the all minus one learning models
        for datasource_pair in self.unique_source_pairs:
            #get record pairs from the labeled set that belong to the current datasource pair
            record_pairs = self.labeled_set[self.labeled_set.datasource_pair != datasource_pair]
            #I can only learn sth if both labels are in the labeled set
            if record_pairs.label.nunique()==2:
                data_subset,labels_subset = self.get_feature_vector_subset(record_pairs)
                model = getClassifier(self.classifier_name, n_estimators=10, random_state=1)
                model.fit(data_subset,labels_subset)
                self.learning_models_all_minus_one[datasource_pair] = model
        
        
    
    def update_learning_models_per_ds(self, boot_state=False):

        #and now update the overall model
        if self.labeled_set.label.nunique()==2:
            
            #update the learning models per group
            if 'mult_models' in self.criteria:
                self.learning_models.update(self.get_mult_models())
            
            all_data, all_labels = self.get_feature_vector_subset(self.labeled_set)
            
            if (not boot_state):
                #increase one tree per iteration
                self.learning_models['boot_all'].n_estimators +=1
                self.learning_models['boot_all'] = self.learning_models['boot_all'].fit(all_data, all_labels)
           
            model = getClassifier(self.classifier_name, random_state=1)
            self.learning_models['all'] = model.fit(all_data, all_labels)
            
            model = getClassifier(self.classifier_name, n_estimators=10)
            self.learning_models['all_simple'] = model.fit(all_data, all_labels)
            
            if self.unlabeled_set['graph_inferred_label'].isnull().values.all():
                self.learning_models['boost_graph'] =  copy.copy(self.learning_models['all'])
            else:
                boost_model = getClassifier(self.classifier_name, random_state=1)
                small_cc_data, small_cc_labels = self.get_feature_vector_from_unlabeled_data(self.unlabeled_set[self.unlabeled_set.graph_cc_size<=self.count_sources])
                data_for_boost = pd.concat([small_cc_data, all_data])
                labels_for_boost = np.concatenate((small_cc_labels, all_labels))
                self.learning_models['boost_graph'] = boost_model.fit(data_for_boost, labels_for_boost)
                
                self.log.pred_graph_diff_log(self)
                self.log.pred_graph_diff_ALL_log(self)
    
    def get_feature_vector_subset(self, labeled_set_subset, getLabels=True):
        
        fv_subset = pd.merge(labeled_set_subset[['source','target']], self.feature_vector, how='left', on=['source','target'])
        data_subset = fv_subset.drop(['source','target', 'label'], axis=1)
        labels_subset = fv_subset['label']
        
        if getLabels: return data_subset,labels_subset.values
        else : return data_subset
    
    def get_feature_vector_from_unlabeled_data(self, unlabeled_set_subset, getLabels=True):
        
        fv_subset = pd.merge(unlabeled_set_subset[['source','target','graph_inferred_label']], self.feature_vector, how='left', on=['source','target'])
        data_subset = fv_subset.drop(['source','target', 'graph_inferred_label','label'], axis=1)
        labels_subset = fv_subset['graph_inferred_label']
        
        if getLabels: return data_subset,labels_subset.values
        else : return data_subset

    
    def evaluateCurrentModel(self, iteration):
        gs_fv_data = self.gs.drop(['source','target', 'label'], axis=1)
        gs_fv_labels = self.gs['label']
        
        gs_y_predict = self.learning_models['all'].predict(gs_fv_data)
        prec, recall, fscore, support  = precision_recall_fscore_support(gs_fv_labels, gs_y_predict, average='binary')
        
        gs_y_predict_boot = self.learning_models['boot_all'].predict(gs_fv_data)
        prec_boot, recall_boot, fscore_boot, support_boot  = precision_recall_fscore_support(gs_fv_labels, gs_y_predict_boot, average='binary')
        
        gs_y_predict_boost_graph = self.learning_models['boost_graph'].predict(gs_fv_data)
        prec_boost_graph, recall_boost_graph, fscore_boost_graph, support_boost_graph  = precision_recall_fscore_support(gs_fv_labels, gs_y_predict_boost_graph, average='binary')
        
        self.results.loc[iteration].P_model=round(prec,3)
        self.results.loc[iteration].R_model=round(recall,3)
        self.results.loc[iteration].F1_model_micro=round(fscore,3)
        self.results.loc[iteration].F1_model_micro_boot=round(fscore_boot,3)
        self.results.loc[iteration].F1_model_micro_boost_graph=round(fscore_boost_graph,3)
        
        #f1 per group based model
        datasource_pair_f1=dict()
        datasource_pair_f1_boost=dict()
    
        all_correct = []
        all_mult_pred = []
        for datasource_pair in self.unique_source_pairs:
            #if datasource_pair in self.learning_models:
            record_pairs = self.gs[self.gs_metadata.datasource_pair == datasource_pair].index
            gs_fv_data = self.gs.loc[record_pairs].drop(['source','target', 'label'], axis=1)
            gs_fv_labels = self.gs.loc[record_pairs]['label']
            #group
            if 'mult_models' in self.criteria:
                group_of_task = self.groups_of_tasks[datasource_pair]
                gs_y_predict_task_based =  self.learning_models[group_of_task].predict(gs_fv_data)
                all_correct = np.concatenate((all_correct, gs_fv_labels))
                all_mult_pred = np.concatenate((all_mult_pred, gs_y_predict_task_based))
            
            #micro
            gs_y_predict = self.learning_models['all'].predict(gs_fv_data)
            prec, recall, fscore, support  = precision_recall_fscore_support(gs_fv_labels, gs_y_predict, average='binary')
            datasource_pair_f1[datasource_pair]=round(fscore,3)
            #micro boost
            gs_y_predict_boost =  self.learning_models['boost_graph'].predict(gs_fv_data)
            prec_boost, recall_boost, fscore_boost, support_boost  = precision_recall_fscore_support(gs_fv_labels, gs_y_predict_boost, average='binary')
            datasource_pair_f1_boost[datasource_pair]=round(fscore_boost,3)

       
        self.results.loc[iteration].F1_pairwise_model=datasource_pair_f1
        self.results.loc[iteration].F1_pairwise_model_boosted=datasource_pair_f1_boost
        self.results.loc[iteration].F1_model_macro=np.mean(list(datasource_pair_f1.values()))
        self.results.loc[iteration].F1_model_macro_boost_graph=np.mean(list(datasource_pair_f1_boost.values()))
        if 'mult_models' in self.criteria:
            self.results.loc[iteration].F1_model_micro_task_based= precision_recall_fscore_support(all_correct, all_mult_pred, average='binary')[2]
        
    def calculateF1ofInformants(self, informants):
        true_labels = self.unlabeled_set_metadata.label

        votes_length = len(informants[0])
        votes_F1 = [0]*votes_length

        results_row = dict()
        for i in range(votes_length):
            votes_of_i = list([v[i] for v in informants])
            prec, recall, fscore, support  = precision_recall_fscore_support(true_labels, votes_of_i, average='binary', zero_division=0)
            votes_F1[i] = fscore

            inf_name = "inf_"+str(i)
            results_row[inf_name]=fscore

        self.informants_eval = self.informants_eval.append(results_row, ignore_index=True)
    
    def count_sources(self):
        i=1
        sum_=0
        while sum_<len(self.unique_source_pairs):
            sum_+=i
            i=i+1
        
        self.count_sources=i
    
    def get_mult_models(self):
          
        learning_models = dict()
        for source in self.rltd_top_setting:             
            record_pairs_train = self.labeled_set[(self.labeled_set.datasource_pair == source)]
                
            train_X, train_y = self.get_feature_vector_subset(record_pairs_train)
            model = getClassifier(self.classifier_name, random_state=1)
            model.fit(train_X,train_y)
            
            group_of_task = self.groups_of_tasks[source]
            learning_models[group_of_task] = model
        return learning_models
        
    def get_heatmap_of_iteration(self):
        xval_scoring = {'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score), 'f1_score' : make_scorer(f1_score)}  
        results_hmap = pd.DataFrame(index=self.unique_source_pairs, columns=self.unique_source_pairs)
        #self_scores = dict()
        
        for source in sorted(list(self.unique_source_pairs)):
            small_cc_train = self.unlabeled_set[(self.unlabeled_set.graph_cc_size<=self.count_sources) & (self.unlabeled_set.datasource_pair == source)]
            
            small_cc_data_train, small_cc_labels_train = self.get_feature_vector_from_unlabeled_data(small_cc_train)
            record_pairs_train = self.labeled_set[self.labeled_set.datasource_pair == source]
            lab_train_X, lab_train_y = self.get_feature_vector_subset(record_pairs_train)
            
            train_X = pd.concat([small_cc_data_train, lab_train_X])
            train_y = np.concatenate((small_cc_labels_train, lab_train_y))
            
            for target in sorted(list(self.unique_source_pairs)):
                
                if (source is target):
                    
                    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
                    model = getClassifier(self.classifier_name, n_estimators=10, random_state=1)
                    model.fit(X_train,y_train)
                    pred = model.predict(X_test).tolist()
                    rec, recall, fscore, support  = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
                    #self_scores[source]=fscore
                    
                    #fscore=-1

                else:
                    
                    small_cc_test = self.unlabeled_set[(self.unlabeled_set.graph_cc_size<=self.count_sources) & (self.unlabeled_set.datasource_pair == target)]
                    small_cc_data_test, small_cc_labels_test = self.get_feature_vector_from_unlabeled_data(small_cc_test)                    
                    record_pairs_test = self.labeled_set[self.labeled_set.datasource_pair == target]
                    lab_test_X, lab_test_y = self.get_feature_vector_subset(record_pairs_test)
                    
                    test_X = pd.concat([small_cc_data_test, lab_test_X])
                    test_y = np.concatenate((small_cc_labels_test, lab_test_y))
                    
                    model = getClassifier(self.classifier_name, n_estimators=10, random_state=1)
                    model.fit(train_X,train_y)
                    pred = model.predict(test_X).tolist()
                    
                    prec, recall, fscore, support  = precision_recall_fscore_support(test_y, pred, average='binary', zero_division=0)
                    #accur = accuracy_score(test_y, pred)
    
                results_hmap.loc[source, target] = fscore
        return results_hmap
    
    def get_best_transf_setting(self, showHeatmap=False):
        correlation_thresh = 0.9
        if self.query_strategy!='almser_group':
            ntl_heatmap = self.get_heatmap_of_iteration()
            ntl_heatmap = ntl_heatmap.reindex(self.rltd.index, columns=self.rltd.columns)
            ntl_heatmap  = ntl_heatmap.apply(pd.to_numeric)

            correlation =statistics.mean(self.rltd.corrwith(ntl_heatmap, axis=0))
            print("Correlation between heatmap and task relatedness scores (per column): ",correlation)
        else: correlation = 0
        #check if the correlate already enough. If yes, use heatmap. If no use relatedness - do not recalculate rltd top setting once it is set
        if correlation<correlation_thresh or self.query_strategy=='almser_group' : 
            print("Based on unsupervised relatedness")
            heatmap_ = self.rltd
            if self.rltd_top_setting!='unassigned':
                return self.rltd_top_setting
    
        else:
            print("Based on supervised naive transfer learning results")
            heatmap_ = ntl_heatmap
        
        if showHeatmap:
            ax = sns.heatmap(heatmap_.to_numpy(dtype=float), xticklabels=heatmap_.index, yticklabels=heatmap_.columns, annot=True)
            plt.show()
        all_combinations = []
        print("Calculate best setting")
        for r in range(len(heatmap_.index) + 1):
            combinations_object = itertools.combinations(heatmap_, r)
            combinations_list = list(combinations_object)
            all_combinations += combinations_list
            
        transferrability_scores = pd.DataFrame(columns=['setting', 'score', 'count_tasks', 'pen_score'])
        #for task in heatmap_.columns:
            #heatmap_.at[task,task] = np.nan
        for task_combi in all_combinations:
            if len(task_combi)==0: continue;
            transf_score = heatmap_.loc[task_combi, :].max(axis=0).mean()
            new_row = {'setting':task_combi, 'score':transf_score, 'count_tasks':len(task_combi), 'pen_score':transf_score-(len(task_combi)*0.01)}
            
            transferrability_scores = transferrability_scores.append(new_row, ignore_index=True)

        #top_setting = transferrability_scores.sort_values('pen_score', ascending=False).head(1)['setting']
        # do not penalize


        if self.query_strategy=='almser_group':
            top_setting = transferrability_scores.sort_values('pen_score', ascending=False).head(1)['setting']
        
        flattened_top = [item for sublist in top_setting for item in sublist]

        flattened_top = self.check_if_enough(flattened_top)
        
        #if correlation<correlation_thresh: self.rltd_top_setting=flattened_top
        
        
        return flattened_top

    def check_if_enough(self, flattened_top):
        print("Check if the setting is enough toreach the quota: ", flattened_top)
        self.unlabeled_set['datasource_pair']
        pairs_fv_train_sub= copy.copy(self.unlabeled_set[self.unlabeled_set.datasource_pair.isin(flattened_top)])
        while (pairs_fv_train_sub.shape[0]<self.quota):
            print("Setting not enough to reach the quota. Will add another task.")

            additional_source = random.choice(list(set(self.unlabeled_set['datasource_pair'])))
            while additional_source in flattened_top:
                additional_source = random.choice(list(set(self.unlabeled_set['datasource_pair'])))
            flattened_top.append(additional_source)
            pairs_fv_train_sub= copy.copy(self.unlabeled_set[self.unlabeled_set.datasource_pair.isin(flattened_top)])
        return flattened_top