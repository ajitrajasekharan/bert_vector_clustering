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

**Step 1c:** Execute run.sh and then choose option 1 to create cluster file debug_pivots.txt

**Step 1d:** Run 

./extract_labels.sh 

to combine user defined labels in map_labels.txt and debug_pivots.txt. 

**Note** these user labels are only applicable to bert_large_cased model clustering. One would have to create a new  map_labels.txt file from debug_pivots.txt for another model


Additionally 

./run.sh 

has options to look at cumulative distribution of a model's words, find pivot among a set of terms, entity labeling,  etc.

# Sample Outputs

Output of clustering using bert_large_cased are in results directory

cum_dist.txt - output of run.sh with option 1

debug_pivots.txt - output of run.sh with option 2

labels.txt and stats_dict.txt - output of running extract_labels.sh


# Steps to run this as a service for downstream unsupervised NER

./run_dist_v2_server.sh 

To test this service 

- wget -O out http://127.0.0.1:8043/"cat dog cow horse"

- cat out
  
  *BIO-SPECIES BIO-SPECIES BIO-SPECIES BIO BIO BIO BIO BIO BIO BIO*
  
  The descriptors "cat dog cow horse" are typically the predictions for a masked word in a sentence. The results are all BIO-SPECIES or BIO capturing the entity type of the masked word. Refer to the medium post link above for more details


# Mask tests

To examine models output prediction scores for each position 

- python all_word_no_mask.py

To examine the cosine values as well as bias values with output vectors *(which leads to the prediction scores displayed by all_word_no_mask.py)*  from the MLM head (or topmost layer)

 - python graph_test.py `pwd`  `pwd`/bert_vectors.txt `pwd`/vocab.txt 1 1

# License

This repository is covered by MIT license. 
