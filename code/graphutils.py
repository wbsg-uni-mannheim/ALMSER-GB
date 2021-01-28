from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import chain
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import Counter
from networkx.algorithms.flow import *
from networkx.algorithms.community import greedy_modularity_communities

    
def getSubgraphOfNode(graph, node):
    return nx.subgraph(graph, nx.node_connected_component(graph,node))

def evaluateCurrentGraph(graph_eval,gs_pairs):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for index, row in gs_pairs.iterrows():
        source_id = row['source']
        target_id = row['target']
        label = row['label']
        
        if graph_eval.has_node(source_id) and graph_eval.has_node(target_id):
            clustering_decision = nx.has_path(graph_eval,source_id,target_id)
        else: clustering_decision=False
        if (clustering_decision and label): true_positives+=1
        elif (clustering_decision and not label): false_positives+=1
        elif (not clustering_decision and label): false_negatives+=1
        elif (not clustering_decision and not label): true_negatives+=1

    p = round(true_positives/(true_positives+false_positives),3)
    r = round(true_positives/(true_positives+false_negatives),3)
    f1= (2*p*r)/(p+r)
    return p,r,f1
    
#draw the graph
def drawGraph(G):
    plt.figure(3,figsize=(15,12)) 
    pos = nx.spring_layout(G)
    nx.draw(G,pos,edge_color='black',node_size=500,node_color='pink',alpha=0.9,linewidths=1,
    labels={node:node for node in G.nodes()})
    edge_labels = nx.get_edge_attributes(G,'capacity')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_color='red')
    plt.axis('off')
    plt.show()

def has_edge_(G, source, target):
    if not (G.has_node(source) and G.has_node(target)): return False
    else: return G.has_edge(source, target)

def has_path_(G, source, target):
    
    if not (G.has_node(source) and G.has_node(target)): return False

    else : return nx.has_path(G, source, target)

#only if at least one of nodes involved has >2 neighbors
def find_bridges_(graph, not_in_list):
    bridges_of_graph = list(nx.bridges(graph))
    bridges_to_be_removed = list(filter(lambda b, G=graph: (len(list(G.neighbors(b[0])))>2 and nx.clustering(G, b[0])>0)  and (len(list(G.neighbors(b[1])))>2 and nx.clustering(G, b[1])>0) , bridges_of_graph))
    bridges_left = [x for x in bridges_to_be_removed if ((x[0],x[1]) not in not_in_list and (x[1],x[0]) not in not_in_list)]
    
    return bridges_left


def constructGraphFromWeightedPredictions(predicted_labels, labeled_set, unlabeled_set, remove_bridges, unlabeled_set_md, sources_count):
    
    predicted_true = unlabeled_set[['source','target','pre_proba']][predicted_labels]

    labeled_true = labeled_set[labeled_set.label][['source','target']]
    labeled_true['pre_proba'] = 100
    labeled_false = labeled_set[labeled_set.label==False][['source','target']]

    predicted_true_pairs = [tuple(x) for x in predicted_true.values]
    labeled_true_pairs = [tuple(x) for x in labeled_true.values]
    labeled_false_pairs = [tuple(x) for x in labeled_false.values]

    G = nx.Graph()
    G.add_weighted_edges_from(predicted_true_pairs, weight='capacity')
    G.add_weighted_edges_from(labeled_true_pairs, weight='capacity')
    G.remove_edges_from(labeled_false_pairs)
    
    
    if (remove_bridges):       
        bridges = find_bridges_(G, labeled_true_pairs)
        G.remove_edges_from(bridges)


    #remove minimun cuts so that there is no path between labeled false pairs
    for lab_false in labeled_false_pairs:
        if has_path_(G, lab_false[0], lab_false[1]):
            cc_of_node = nx.node_connected_component(G, lab_false[0])
            G_of_cc = G.subgraph(cc_of_node)
            cut_weight, partitions = nx.minimum_cut(G_of_cc, lab_false[0], lab_false[1])
            
            edge_cut_list = [] # Computed by listing edges between the 2 partitions
            for p1_node in partitions[0]:
                for p2_node in partitions[1]:
                    if G_of_cc.has_edge(p1_node,p2_node):
                        edge_cut_list.append((p1_node,p2_node))
                        cut_label = unlabeled_set_md[((unlabeled_set_md.source==p1_node) & (unlabeled_set_md.target==p2_node)) |  ((unlabeled_set_md.source==p2_node) & (unlabeled_set_md.target==p1_node))].label
                        
                        print("Cut label", cut_label)
            G.remove_edges_from(edge_cut_list)
    
  
    return G

def constructGraphFromPredictions(predicted_labels, labeled_set, unlabeled_set, remove_bridges, unlabeled_set_md):
    
    predicted_true = unlabeled_set[['source','target']][predicted_labels]

    labeled_true = labeled_set[labeled_set.label][['source','target']]
    labeled_false = labeled_set[labeled_set.label==False][['source','target']]

    predicted_true_pairs = [tuple(x) for x in predicted_true.values]
    labeled_true_pairs = [tuple(x) for x in labeled_true.values]
    labeled_false_pairs = [tuple(x) for x in labeled_false.values]

    G = nx.Graph()
    G.add_edges_from(predicted_true_pairs)
    G.add_edges_from(labeled_true_pairs)
    G.remove_edges_from(labeled_false_pairs)
    
    
    if (remove_bridges):       
        bridges = find_bridges_(G, labeled_true_pairs)
        G.remove_edges_from(bridges)


    #remove cuts so that there is no path between labeled false pairs
    for lab_false in labeled_false_pairs:
        if has_path_(G, lab_false[0], lab_false[1]):
        
            cuts = nx.minimum_edge_cut(G, s=lab_false[0], t=lab_false[1], flow_func=shortest_augmenting_path)
            G.remove_edges_from(cuts)
        
    
    return G
