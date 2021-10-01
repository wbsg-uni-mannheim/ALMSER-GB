import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import multiprocessing
from contextlib import contextmanager
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from multiprocessing import Pool

def labeled_set_init():
    return pd.DataFrame(columns=['source','target','datasource_pair','label','predicted_label','disagreement','graph_inferred_label', 'inf_score', 'votes', 'sel_proba','cc_size','task_rltd'])


def unlabeled_set_init():
    return pd.DataFrame(columns=['source','target','datasource_pair','votes', 'disagreement', 'datasource_pair_frequency','inf_score','graph_inferred_label', 'predicted_label','graph_cc_size', 'sel_proba','task_rltd' ])

def results_init(no_rows):
    return pd.DataFrame(index=np.arange(no_rows), columns=["P_model","R_model","F1_model_micro", "F1_model_macro","F1_model_micro_boot","F1_model_micro_boost_graph", "F1_model_macro_boost_graph","F1_pairwise_model","F1_model_micro_task_based","Query","Query_Label","Strategy","Labeled_Set_Size"]) 

def get_criteria_list(query_strategy):
    if query_strategy =='random':  return []
    elif query_strategy =='random_stratified':  return ['datasource_pair_frequency']
    elif query_strategy =='betweeness':  return ['edge_betweeness']
    
    elif query_strategy == 'disagreement' : return ['disagreement','predicted_label']
    
    elif query_strategy == 'exploit_explore' : return ['disagreement','predicted_label','ensemble_graph','datasource_pair_frequency']
    
    elif query_strategy == 'rltd_disagreement' : return ['disagreement','predicted_label','task_rltd']
    
    elif query_strategy =='disagreement_stratified':  return ['disagreement','datasource_pair_frequency','predicted_label']

    elif query_strategy=='disagreement_post_graph' : return ['disagreement','predicted_label','ensemble_graph']
    
    elif query_strategy=='graph_based' : return ['predicted_label','ensemble_graph']
    
    elif query_strategy=='greedy' : return ['greedy_inf_gain_search']
    
    elif query_strategy=='almser_gb': return ['predicted_label','ensemble_graph','disagreement_graph_pred']
    
    elif query_strategy=='almser_gb_transfer': return ['predicted_label','ensemble_graph','disagreement_graph_pred', 'datasource_pair_frequency', 'disagreement']
    
    elif query_strategy=='almser_gb_explore_exploit': return ['predicted_label','ensemble_graph','disagreement_graph_pred', 'datasource_pair_frequency', 'disagreement']
    
    elif query_strategy=='uncertainty': return ['uncertainty']
    elif query_strategy=='margin_boost_learner': return ['uncertainty','predicted_label','ensemble_graph']
    elif query_strategy=='almser_group': return ['predicted_label','ensemble_graph','disagreement_graph_pred', 'datasource_pair_frequency', 'disagreement']

def disagreement_score(votes):
    non_AL_model_votes = votes[1:]
    AL_model_vote = votes[0]
    if len(set(non_AL_model_votes))==1:
        if AL_model_vote==non_AL_model_votes[0]: return 0
        else: return 1
    else: 
        score = 1 - max(votes.count(True),votes.count(False))/len(votes)
        return score

def gainAfterCandidateX(idx,  almser_exp, all_data, all_labels, cur_fscore, gs_fv_data, gs_fv_labels):
    
    fv_pair,  fv_pair_label= almser_exp.get_feature_vector_subset(almser_exp.unlabeled_set.loc[[idx],:], getLabels=True)
    concat_data = pd.concat([all_data, fv_pair])
    concat_label = np.concatenate((all_labels, fv_pair_label))

    model_w_pair = RandomForestClassifier(random_state=1)
    model_w_pair.fit(concat_data, concat_label)
    model_pred = model_w_pair.predict(gs_fv_data)
    pair_fscore  = precision_recall_fscore_support(gs_fv_labels, model_pred, average='binary')[2]
    inf_gain = pair_fscore-cur_fscore
    return (idx, inf_gain)        

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def run_apply_async_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm

def run_imap_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm    