# ALMSER-GB

This project contains the code and data necessary for reproducing the results of the paper 
<b>"Graph-boosted Active Learning for Multi-Source Entity Resolution"</b> by Anna Primpeli and Christian Bizer submitted for ISWC2021.

<b>INSTALLATION OF REQUIRED PACKAGES:</b>
1. The project runs on Anaconda 4.6.8 with Python 3.7 without further package requirements.
2. If you do not have an Anaconda distribution, please install the packages found in the requirements.txt file.

<b>HOW TO RUN AN ALMSER EXPERIMENT?</b>
<ol>
<li>Navigate to the ALMSER_Experiments.ipynb</li>
<li>Configure the first cell *Specifications* by:
  <ul>
  <li>Defining the path where the feature vector files are stored, the output path, and splitter character
  which is used on the naming convention of the feature vector files to split the source names (e.g. the feature vector 2_1.csv contains record pairs from sources 2 and 1
  and the splitter character is '_')</li>
  
  <li>Defining the max queries (stopping criterion for Active Learning), #runs of the experiment (if >1 the results will be averaged and the std will be calculated),
  the query strategy (choose from: almser_gb, almser_gb_group, disagreement (HeALer) (B1-QHC), uncertainty (B2-MB))</li>
  </ul>
</li> 
<li>(optional) Run the second cell *Passive Learning Results* to get the Precision, Recall and F1 score when all available training data is used.</li>
<li>Run the third cell *Load the stored files and start ALMSER* to start ALMSER based on the configuration you set on the first cell.</li>
<li>(optional) Run the fourth cell *Store Results* if you wish to save the results of your experiment together with some log files in the output path.</li>
</ol>

The rest of the python notebooks provide functionalities for preparing and profiling the data:
1. ALMSER_CC_split: splits the record pairs of each multi-source ER task into pool and test, considering the connected components of the complete graph.
It ensures that there is no leakage by graph transitivity from the pool set to the test set.
2. ALMSER_Plotting: plots the results of the AL experiments. 
3. MatchingTaskProfiling: Profiles the matching tasks along different EM task profiling dimensions, such as density, corner cases and important features.


