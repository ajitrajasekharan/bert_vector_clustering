# bert_vector_clustering
Clustering learned BERT vectors for downstream tasks like unsupervised NER, unsupervised sentence embeddings etc.

**Unsupervised training of BERT yields a model and context insenstive  word vectors. These word vectors are stored in pytorch_model.bin. This repository has simple utilities to extract those vectors, cluster them, etc.**

[Medium post that uses this repository tools to cluster BERT's context insensitive word vectors for unsupervised NER](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)


# Steps to cluster

**Step 1a:**
	Fetch a model and its vocab file using fetch_model.sh
	./fetch_model.sh
	This will download bert_large_cased and its vocab for testing. *Refer to huggingface model repositories for other model URLs.*

**Step 1b:**
	Extract vectors from the model
	python examine_model.py 2 > bert_vectors.txt	

**Step 1c:** Execute run.sh and then choose option 1 to create cluster file debug_pivots.txt

**Step 1d:** Run extract_labels.sh to combine user defined labels in map_labels.txt and debug_pivots.txt. 
**Note** these user labels are only applicable to bert_large_cased model clustering. One would have to create a new  map_labels.txt file from debug_pivots.txt for another model


Additionally run.sh (dist_v2.py)  has options to look at cumulative distribution of a model's words, find pivot among a set of terms, entity labeling,  etc.

# Sample Outputs

Output of clustering using bert_large_cased are in results directory

cum_dist.txt - output of run.sh with option 1

debug_pivots.txt - output of run.sh with option 2

labels.txt and stats_dict.txt - output of running extract_labels.sh




# License

This repository is covered by MIT license. 
