# bert_vector_clustering
Clustering learned BERT vectors for downstream tasks like unsupervised NER, unsupervised sentence embeddings etc.

*Unsupervised training of BERT yields a model and context insenstive  word vectors. These word vectors are stored in pytorch_model.bin. This repository has simple utilities to extract those vectors, cluster them, etc.



#Steps to cluster

*Step 1a:*
	Fetch a model and its vocab file using fetch_model.sh
	./fetch_model.sh
	This will download bert_large_cased and its vocab

*Step 1b:*
	Extract vectors from the model
	python examine_model.py 2 > bert_vectors.txt	

