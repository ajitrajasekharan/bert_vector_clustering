# bert_vector_clustering
Clustering learned BERT vectors for downstream tasks like unsupervised NER, unsupervised sentence embeddings etc.

**Unsupervised training of BERT yields a model and context insenstive  word vectors. These word vectors are stored in pytorch_model.bin. This repository has simple utilities to extract those vectors, cluster them, etc.**

[Medium post that uses this repository tools to cluster BERT's context insensitive word vectors for unsupervised NER](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)


# Steps to cluster

**Step 1a:**
	Fetch a model and its vocab file using fetch_model.sh
	
	./fetch_model.sh
	
	This will download bert_large_cased and its vocab for testing.
	
	** Refer to huggingface model repositories for other model URLs. **

**Step 1b:**
	Extract vectors from the model
	
	python examine_model.py 2 > bert_vectors.txt	

**Step 1c:** Execute run.sh and then choose option 1 to create cluster file adaptive_debug_pivots.txt. *If the intent is to do NER for custom entity types, replace bootstrap_entities.txt with the entity types of the subset of the terms of interest present in the  vocabulary of the model being used for NER. Update bootstrap_entities.txt file before clutering*

**Step 1d:** Run 

./extract_labels.sh 

*This generate labels.txt and stats_dict.txt which is copied to results directory for use in the NER service mentioned below.*


Additionally 

./run.sh 

has options to look at cumulative distribution of a model's words, find pivot among a set of terms, entity labeling,  etc.

# Sample Outputs

Output of clustering using bert_large_cased are in results directory

cum_dist.txt - output of run.sh with option 1

adaptive_debug_pivots.txt - output of run.sh with option 2

labels.txt and stats_dict.txt - output of running extract_labels.sh


# Steps to run this as a service for downstream unsupervised NER

./run_dist_v2_server.sh 

To test this service 

- wget -O out http://127.0.0.1:8043/"cat dog cow horse"

- cat out
  
  *BIO-SPECIES 13 BIO-SPECIES 13 BIO-SPECIES 13*
  
  The descriptors "cat dog cow horse" are typically the predictions for a masked word in a sentence. The results are all BIO-SPECIES or BIO capturing the entity type of the masked word. Refer to the medium post link above for more details


# Mask tests

To examine models output prediction scores for each position 

- python all_word_no_mask.py

To examine the cosine values as well as bias values with output vectors *(which leads to the prediction scores displayed by all_word_no_mask.py)*  from the MLM head *(or topmost layer)*. To examine from MLM head PyTorch code would need to be patched as explained in this [post](https://towardsdatascience.com/swiss-army-knife-for-unsupervised-task-solving-26f9acf7c023?source=friends_link&sk=6d4bc39010d8026d4bf1a394a90c08f3)

 - python graph_test.py `pwd`  `pwd`/bert_vectors.txt `pwd`/vocab.txt 1 1


# Revision notes

17 Sept 2021

The original labeling of vocab file terms was reduced by just labeling cluster pivots by manually looking at the clusters the pivots belong to. For example PERSON or PERSON/LOCATION etc.


In the new approach a bootstrap list of terms labeled by humans are used to automatically label clusters.
In the new approach not all the vocab terms are labeled. Some percentage is (and this can grow over time as we create new vocabs)
The labeled subset is used to label other terms in a cluster.

So the new approach is, 
1) For each term,when clustering, capture the histogram of label elements across clusters. 
2) Then for each term in a cluster, aggregate the number of times an entity is mentioned for each term that is present in the bootstrap list **only**.  
3) For terms in the bootstrap list and part of vocab, but not present in cluster, just pick the manual label for term with 0 counts.
4) For terms that are not in bootstrap cluster, separately aggregate an inferred list. At the end this is sorted and manually examined for further manual labeling. This labeling process is assisted by the inferred entity labels for a term by virtue of the clusters it is part of.  
5) Output the entire vocab file with entity info for each term. Terms that occurred in bootsrap file and in clusters will have entities reordered by the clustering process. Terms that occured in bootstrap file but not in clusters will inherit manual labels as is without any reordering or count information. Terms tha did not occurr in bootstrap file and showed up in clusters will be output as an inferred entities list for futher manual labeling (at least the top frequence ones). Terms that did not occurr in bootstrap and and did not occur in clusters, but just present in vocab will be tagged "OTHER".
6) Note while bootstrap entities are case insensitive, clustering is on case sensitive terms. So even if entities for a cased and uncased version of a term are present together in bootstrap file, they separate out in output with different ordering of the entities based on the clusters they occur. Example is eGFR and EGFR.
For instance, if the bootstrap list order for the case insenstive version of egfr is

 - GENE/LAB_PROCEDURE/PROTEIN/RECEPTOR egfr

after clustering (in this case vocab contains eGFR and EGFR) the cased variations separate and have different orders and cluster counts
 - GENE/PROTEIN/LAB_PROCEDURE/RECEPTOR 134/36/23/10 EGFR

 - LAB_PROCEDURE/GENE/PROTEIN/RECEPTOR 8/7/5/3 eGFR

In essence clustering reorders the manuaally labeled entities for a term into the different context independent meanings of the term, based on the cased versions of the term in the vocabulary.

# bootstrap labeling

When starting in a new domain with no labels, start with an empty labels.txt and bootstrap_entities.txt.  Cluster (run.sh with option 1 folllowed by 0) and then examine cluster pivots to label them. Then rerun clustering and select candidates from inferred.txt. Add this to bootstrap_entities.txt list and repeat. 

Note the bootstrap labeling addresses the labeling of terms that appear in clusters. However, there may be terms that are not clustered (singletones/empty). It might be worthwhile labeling at least some of these manually since they could appear in run time contexts as neigbors. In practice, we potneitally could for the most part get away without this, but it is worth keeping this mind to improve performance say for unsupervised NER.

# License

This repository is covered by MIT license. 
