from ALMSER import *
from collections import Counter 
from itertools import chain
from networkx.algorithms import community
from pandas import Series
import pdb

class ALMSER_EXP(ALMSER):    
    pass

    def run_AL(self):
        print("Start ALMSER")
        
        print_progress(1, self.quota, prefix="ALMSER Mode: Active Learning")
        initial_qs = self.query_strategy
        initial_criteria = self.criteria
        for i in range(self.quota):           

            #do the basic for first iterations
            if  ('graph_signal' in initial_criteria) or ('ensemble_graph' in initial_criteria):
                if i>20:
                    self.query_strategy=initial_qs
                else: 
                    self.query_strategy='disagreement'

            self.criteria = get_criteria_list(self.query_strategy)
            self.update_criteria_scores(i)
            self.get_informativeness_score()
            candidate,strategy = self.select_pair()

            s_id, t_id, true_label = self.update_after_answer(candidate)
                        
            self.results.loc[i].Query = s_id+"-"+t_id
            self.results.loc[i].Query_Label = true_label
            self.results.loc[i].Strategy = strategy
            self.results.loc[i].Labeled_Set_Size = self.labeled_set.shape[0]

            if 'all' in self.learning_models:
                self.evaluateCurrentModel(i)

            print_progress(i+1, self.quota, prefix="ALMSER Mode: Active Learning")
           
            
    def update_criteria_scores(self, iteration):
        try: 
            if 'predicted_label' in self.criteria:
                unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
                self.unlabeled_set['predicted_label'] = self.learning_models['all'].predict(unlabeled_data)
                pre_proba_both_classes= self.learning_models['all'].predict_proba(unlabeled_data)
                max_proba = map(lambda x : max(x), pre_proba_both_classes)
                self.unlabeled_set['pre_proba'] = list(max_proba)
   
            
            if 'disagreement' in self.criteria:
                if self.labeled_set.label.nunique()==2:
                    unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
                    dis, votes = self.calculate_disagreement(unlabeled_data, ('graph_signal' in self.criteria), ('correct_prediction' in self.criteria))
                    self.unlabeled_set['disagreement']=dis
                    self.unlabeled_set['votes']=votes
            
            if 'ensemble_graph' in self.criteria:
                predictions = self.unlabeled_set['predicted_label']
                self.unlabeled_set['majority_vote'] = predictions
                self.G = constructGraphFromWeightedPredictions(predictions, self.labeled_set, self.unlabeled_set, True, self.unlabeled_set_metadata, self.count_sources)
                self.calculate_graph_info()


                self.unlabeled_set['graph_cc_size'] = self.unlabeled_set.apply(lambda row, G=self.G: len(nx.node_connected_component(G, row.source)) if has_path_(G, row.source, row.target) else 0, axis=1)

                self.unlabeled_set['sel_proba'] = self.unlabeled_set.apply(lambda row, G=self.G: 1/row.graph_cc_size if row.graph_inferred_label else 1.0, axis=1)

            
            if 'datasource_pair_frequency' in self.criteria:
                if (self.labeled_set.shape[0]>0) : 
                    history_status = self.get_labeledset_current_status()

                    for ds_p in history_status.keys():
                        ds_p_inx = self.unlabeled_set[self.unlabeled_set.datasource_pair==ds_p].index
                        self.unlabeled_set.loc[ds_p_inx, 'datasource_pair_frequency']=history_status.get(ds_p)

        except Exception as e:
            print(str(e))
            import pdb;pdb.set_trace();
        
    def get_informativeness_score(self):
        
        self.unlabeled_set['inf_score']= 0
        
        if self.query_strategy=="disagreement_stratified" or self.query_strategy=="disagreement_graph_stratified":
            
            self.unlabeled_set['inf_score'] = (self.unlabeled_set['disagreement']*(1-self.unlabeled_set['datasource_pair_frequency']))
               

        elif self.query_strategy=='disagreement' or self.query_strategy=='disagreement_post_graph':
            self.unlabeled_set['inf_score'] = self.unlabeled_set['disagreement']
              
        elif self.query_strategy=='random_stratified':    
            self.unlabeled_set['inf_score'] = 1-self.unlabeled_set['datasource_pair_frequency']

        else:  
            print("Unknown query strategy. Cannot calculate informativeness score.")
            import pdb;pdb.set_trace();
    
    def select_pair(self):
        if self.query_strategy!='random' and not self.unlabeled_set['inf_score'].isnull().all():
            max_qs_score =  self.unlabeled_set['inf_score'].max()
            all_max_candidates = self.unlabeled_set[self.unlabeled_set['inf_score']==max_qs_score]
            candidate = random.choice(all_max_candidates.index)
            
            if self.query_strategy=='disagreement_post_graph':
                #do some post processing. pick with probability 50% what is corrected by the graph. 
                pick_conflict = random.uniform(0, 1)>0.5
                
                if pick_conflict:                
                    sel_proba = all_max_candidates.apply(lambda x: x['sel_proba'] if x['majority_vote']!=x['graph_inferred_label'] else 0, axis=1)
                    candidate = random.choices(all_max_candidates.index, weights=sel_proba,k=1)[0]

                else:
                    candidate = random.choices(all_max_candidates.index,k=1)[0]
                
               
                
            strategy = self.query_strategy
           
                   
        else: 
            #random candidate if selection strategy is null or if it cannot be calculated otherwise
            candidate = random.choice(self.unlabeled_set.index)
            strategy = 'random'
        
        return candidate,strategy
    