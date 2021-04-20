from ALMSER_utils import *
from ALMSER import *

class ALMSER_log(object):    
    
    def __init__(self,quota):
        log_columns = ['addit_train_data_size','addit_train_data_acc','addit_train_data_f1', 'all_graph_acc', 'all_graph_f1', 'cc_distribution', 'small_cc', 'large_cc', 'bridges', 'wrong_bridges', 'cuts', 'wrong_cuts', 'heatmap','correct_graph_from_dis_match',
'correct_graph_from_dis_non_match', 'wrong_graph_from_dis_match', 'wrong_graph_from_dis_non_match']
        self.log_info = pd.DataFrame(index=np.arange(quota), columns=log_columns)
    
    def graph_log(self, almser_obj):
        prec, recall, fscore, support  = precision_recall_fscore_support(almser_obj.unlabeled_set_metadata.label, almser_obj.unlabeled_set.graph_inferred_label, average='binary')
        accuracy = accuracy_score(almser_obj.unlabeled_set_metadata.label, almser_obj.unlabeled_set.graph_inferred_label)
        
        self.log_info.loc[almser_obj.iteration].all_graph_f1=round(fscore,3)
        self.log_info.loc[almser_obj.iteration].all_graph_acc=round(accuracy,3)
        
    def boost_data_log(self, almser_obj):
        small_cc_data, small_cc_labels = almser_obj.get_feature_vector_from_unlabeled_data(almser_obj.unlabeled_set[almser_obj.unlabeled_set.graph_cc_size<=almser_obj.count_sources])
        
        idx_cc_small  =almser_obj.unlabeled_set.index[almser_obj.unlabeled_set.graph_cc_size<=almser_obj.count_sources].tolist()
        true_labels_of_small_cc = almser_obj.unlabeled_set_metadata.loc[idx_cc_small, 'label']
        prec, recall, fscore, support  = precision_recall_fscore_support(true_labels_of_small_cc, almser_obj.unlabeled_set.loc[idx_cc_small, 'graph_inferred_label'], average='binary', zero_division=0)
        accuracy = accuracy_score(true_labels_of_small_cc, almser_obj.unlabeled_set.loc[idx_cc_small, 'graph_inferred_label'])

        self.log_info.loc[almser_obj.iteration].addit_train_data_size = len(idx_cc_small)
        self.log_info.loc[almser_obj.iteration].addit_train_data_acc = accuracy
        self.log_info.loc[almser_obj.iteration].addit_train_data_f1 = fscore

      
    def pred_graph_diff_log(self, almser_obj):
        
        idx_cc_small  =almser_obj.unlabeled_set.index[almser_obj.unlabeled_set.graph_cc_size<=almser_obj.count_sources].tolist()
        
        graph_inferred_labels = almser_obj.unlabeled_set.loc[idx_cc_small, 'graph_inferred_label']
        predicted_labels = almser_obj.unlabeled_set.loc[idx_cc_small, 'predicted_label']
        true_labels_of_small_cc = almser_obj.unlabeled_set_metadata.loc[idx_cc_small, 'label']

        disagreements = 0      
        correct_graph_from_dis=0
        correct_graph_from_dis_match =0
        correct_graph_from_dis_non_match =0
        wrong_graph_from_dis_match =0
        wrong_graph_from_dis_non_match=0
        #how many of the disagreements, have a correc graph inferred label
        for x in list(zip(graph_inferred_labels,predicted_labels,true_labels_of_small_cc)):
            if x[0]!=x[1]:
                disagreements+=1
                if x[0]==x[2]:
                    correct_graph_from_dis+=1
                    if x[2]==True:
                        correct_graph_from_dis_match +=1
                    if x[2]==False:
                        correct_graph_from_dis_non_match +=1
                else:
                    if x[2]==True:
                        wrong_graph_from_dis_match +=1
                    if x[2]==False:
                        wrong_graph_from_dis_non_match +=1
                    
        #print("Additional training data : %i" %len(idx_cc_small))
        #print("Disagreements between graph and predicted labels: %i (%f) of which graph-labels correct: %i" %(disagreements,round((disagreements/len(idx_cc_small)), 3), correct_graph_from_dis))
        
        #print("correct_graph_from_dis_match: %i" %correct_graph_from_dis_match)
        #print("correct_graph_from_dis_non_match: %i" %correct_graph_from_dis_non_match)
        #print("wrong_graph_from_dis_match: %i" %wrong_graph_from_dis_match)
        #print("wrong_graph_from_dis_non_match: %i" %wrong_graph_from_dis_non_match)

        self.log_info.loc[almser_obj.iteration].correct_graph_from_dis_match = correct_graph_from_dis_match
        self.log_info.loc[almser_obj.iteration].correct_graph_from_dis_non_match = correct_graph_from_dis_non_match
        self.log_info.loc[almser_obj.iteration].wrong_graph_from_dis_match = wrong_graph_from_dis_match
        self.log_info.loc[almser_obj.iteration].wrong_graph_from_dis_non_match = wrong_graph_from_dis_non_match

    def pred_graph_diff_ALL_log(self, almser_obj):
                
        graph_inferred_labels = almser_obj.unlabeled_set['graph_inferred_label']
        predicted_labels = almser_obj.unlabeled_set['predicted_label']
        true_labels_of_all_cc = almser_obj.unlabeled_set_metadata['label']

        disagreements = 0      
        correct_graph_from_dis=0
        #how many of the disagreements, have a correc graph inferred label
        for x in list(zip(graph_inferred_labels,predicted_labels,true_labels_of_all_cc)):
            if x[0]!=x[1]:
                disagreements+=1
                if x[0]==x[2]:
                    correct_graph_from_dis+=1
        #print("Disagreements between ALL graph and predicted labels: %i of which graph-labels correct: %i" %(disagreements, correct_graph_from_dis))
                
    
    def cc_distribution_log(self, almser_obj):
        con_components = list(nx.connected_components(almser_obj.G))
        con_components_lengths = [len(x) for x in con_components]
        cc_distribution = Counter(con_components_lengths)
        self.log_info.loc[almser_obj.iteration].cc_distribution = cc_distribution
        count_small_cc = 0
        count_large_cc = 0
        for key in cc_distribution.keys():
            if key<=almser_obj.count_sources:
                count_small_cc+=cc_distribution[key]
            else:
                count_large_cc+=cc_distribution[key]
        
        self.log_info.loc[almser_obj.iteration].small_cc = count_small_cc
        self.log_info.loc[almser_obj.iteration].large_cc = count_large_cc

        
    def bridges_log(self, bridges, almser_obj):
        self.log_info.loc[almser_obj.iteration].bridges = len(bridges)
        wrong_bridges = 0
        for bridge in bridges:        
            true_label = almser_obj.unlabeled_set_metadata[((almser_obj.unlabeled_set_metadata.source==bridge[0]) & (almser_obj.unlabeled_set_metadata.target==bridge[1])) |  ((almser_obj.unlabeled_set_metadata.source==bridge[0]) & (almser_obj.unlabeled_set_metadata.target==bridge[1]))].label
            if len(true_label)>0 : 
                true_label = true_label.values[0]
                if true_label: wrong_bridges+=1
        
        self.log_info.loc[almser_obj.iteration].wrong_bridges = wrong_bridges
       
    def cuts_log(self, cuts, almser_obj):
        self.log_info.loc[almser_obj.iteration].cuts = len(cuts)
        wrong_cuts = 0
        for cut in cuts:
            true_label = almser_obj.unlabeled_set_metadata[((almser_obj.unlabeled_set_metadata.source==cut[0]) & (almser_obj.unlabeled_set_metadata.target==cut[1])) |  ((almser_obj.unlabeled_set_metadata.source==cut[0]) & (almser_obj.unlabeled_set_metadata.target==cut[1]))].label
            if len(true_label)>0 : 
                true_label = true_label.values[0]
                if true_label: wrong_cuts+=1
                
        self.log_info.loc[almser_obj.iteration].wrong_cuts = wrong_cuts
        
    def heatmap_log(self, almser_obj):
        self.log_info.loc[almser_obj.iteration].heatmap = almser_obj.heatmap.values
        display(almser_obj.heatmap)