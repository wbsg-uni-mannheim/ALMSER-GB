import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def labeled_set_init():
    return pd.DataFrame(columns=['source','target','datasource_pair','label','predicted_label','disagreement','graph_inferred_label', 'inf_score', 'votes', 'sel_proba','cc_size'])


def unlabeled_set_init():
    return pd.DataFrame(columns=['source','target','datasource_pair','votes', 'disagreement', 'datasource_pair_frequency','inf_score','graph_inferred_label', 'predicted_label','graph_cc_size', 'sel_proba' ])

def results_init(no_rows):
    return pd.DataFrame(index=np.arange(no_rows), columns=["P_model","R_model","F1_model_micro", "F1_model_macro","F1_model_micro_boot","F1_model_micro_boost_graph", "F1_model_macro_boot","F1_pairwise_model","Query","Query_Label","Strategy","Labeled_Set_Size"]) 

def get_criteria_list(query_strategy):
    if query_strategy =='random':  return []
    elif query_strategy =='random_stratified':  return ['datasource_pair_frequency']
    elif query_strategy =='betweeness':  return ['edge_betweeness']
    elif query_strategy == 'disagreement' : return ['disagreement','predicted_label']
    
    elif query_strategy =='disagreement_stratified':  return ['disagreement','datasource_pair_frequency','predicted_label']

    elif query_strategy=='disagreement_post_graph' : return ['disagreement','predicted_label','ensemble_graph']


def disagreement_score(votes):
    non_AL_model_votes = votes[1:]
    AL_model_vote = votes[0]
    if len(set(non_AL_model_votes))==1:
        if AL_model_vote==non_AL_model_votes[0]: return 0
        else: return 1
    else: 
        score = 1 - max(votes.count(True),votes.count(False))/len(votes)
        return score

