### bert_vector_clustering
Clustering  BERT vocab vectors for downstream tasks like self-supervised [NER](https://github.com/ajitrajasekharan/unsupervised_NER.git) , sentence embeddings etc.

_Self-supervised training of BERT yields a model and context insenstive  word vectors. These word vectors are stored in pytorch_model.bin. This repository has simple utilities to extract those vectors, cluster them for NER._

[Notebook for fill mask prediction](https://colab.research.google.com/github/ajitrajasekharan/bert_vector_clustering/blob/master/test_notebook.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajitrajasekharan/bert_vector_clustering/blob/master/test_notebook.ipynb) 



[Medium post that uses this repository tools to cluster BERT's context insensitive word vectors for unsupervised NER](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)


### Steps to cluster

**Step 1a:** Fetch a model and its vocab file using fetch_model.sh
	
	./fetch_model.sh
	

_This will download bert_large_cased and its vocab for testing. Cased models are preferred for self-supervised NER since they can distinguish terms like eGFR from EFGR._
	
_Refer to huggingface model repository for other model URLs._

**Step 1b:** Extract vectors from the model
	
	python examine_model.py 2 > bert_vectors.txt	

**Step 1c:** Cluster BERT vocab vectors and generate entity signatures
	
	Execute run.sh with option 6


_The output of this is bootstrap_entities.txt with entity signature generated for each term in the vocabulary including subwords. This can then be used for self-supervised [NER](https://github.com/ajitrajasekharan/unsupervised_NER.git) 


Additionally 

	./run.sh 

has options to look at cumulative distribution of a model's words, find pivot among a set of terms, entity labeling,  etc.

### Sample Outputs

Output of clustering using bert_large_cased are in results directory

cum_dist.txt - output of run.sh with option 1

adaptive_debug_pivots.txt - output of run.sh with option 2

labels.txt and stats_dict.txt - output of running extract_labels.sh


### Steps to run this as a service for downstream unsupervised NER

./run_dist_v2_server.sh 

To test this service 

- wget -O out http://127.0.0.1:8043/"cat dog cow horse"

- cat out
  
  *BIO-SPECIES 13 BIO-SPECIES 13 BIO-SPECIES 13*
  
  The descriptors "cat dog cow horse" are typically the predictions for a masked word in a sentence. The results are all BIO-SPECIES or BIO capturing the entity type of the masked word. Refer to the medium post link above for more details


### Mask tests

To examine models output prediction scores for each position 

- python all_word_no_mask.py

To examine the cosine values as well as bias values with output vectors *(which leads to the prediction scores displayed by all_word_no_mask.py)*  from the MLM head *(or topmost layer)*. To examine from MLM head PyTorch code would need to be patched as explained in this [post](https://towardsdatascience.com/swiss-army-knife-for-unsupervised-task-solving-26f9acf7c023?source=friends_link&sk=6d4bc39010d8026d4bf1a394a90c08f3)

 - python graph_test.py `pwd`  `pwd`/bert_vectors.txt `pwd`/vocab.txt 1 1


### Revision notes

**Jan 2022**

This release magnifies the bootstrap labeling using clusterin and also includes subword labeling.

The bootstrap file sample after clustering looks like this 

GENE/PROTEIN/ENZYME/DRUG/PROTEIN_FAMILY/MOUSE_GENE/DISEASE/RECEPTOR/BIO_MOLECULE/LAB_PROCEDURE/MEASURE/MOUSE_PROTEIN_FAMILY/CELL_LINE/CELL/VIRUS/THERAPEUTIC_OR_PREVENTIVE_PROCEDURE/LOCATION/VIRAL_PROTEIN/METABOLITE/CHEMICAL_SUBSTANCE/ORGANIZATION/HAZARDOUS_OR_POISONOUS_SUBSTANCE/UNTAGGED_ENTITY/HORMONE/SURGICAL_AND_MEDICAL_PROCEDURES/CELL_COMPONENT/PERSON/DIAGNOSTIC_PROCEDURE/BACTERIUM/CELL_FUNCTION/BODY_PART_OR_ORGAN_COMPONENT/PHYSIOLOGIC_FUNCTION/ESTABLISHED_PHARMACOLOGIC_CLASS/MEDICAL_DEVICE/ORGAN_OR_TISSUE_FUNCTION/LAB_TEST_COMPONENT/BIO/SPECIES/CONGENITAL_ABNORMALITY/DRUG_ADJECTIVE/TIME/NUCLEOTIDE_SEQUENCE/MENTAL_OR_BEHAVIORAL_DYSFUNCTION/BODY_SUBSTANCE/CELL_OR_MOLECULAR_DYSFUNCTION/LEGAL/DISEASE_ADJECTIVE/SEQUENCE/CHEMICAL_CLASS/VITAMIN/DEVICE/PRODUCT/GENE_EXPRESSION_ADJECTIVE 634/225/175/165/148/147/141/126/99/73/54/44/44/36/30/27/24/24/20/19/19/19/18/15/14/10/9/9/9/7/7/6/6/5/4/4/4/3/3/2/2/2/2/1/1/1/1/1/1/1/1/1/1 EGFR

DISEASE/GENE/UNTAGGED_ENTITY/LAB_PROCEDURE/MEASURE/PROTEIN/DRUG/RECEPTOR/DIAGNOSTIC_PROCEDURE/THERAPEUTIC_OR_PREVENTIVE_PROCEDURE/METABOLITE/SURGICAL_AND_MEDICAL_PROCEDURES/ORGAN_OR_TISSUE_FUNCTION/ORGANIZATION/CHEMICAL_SUBSTANCE/BODY_PART_OR_ORGAN_COMPONENT/LAB_TEST_COMPONENT/DISEASE_ADJECTIVE/ENZYME/PROTEIN_FAMILY/CELL_LINE/MOUSE_GENE/LOCATION/VIRUS/PERSON/MEDICAL_DEVICE/ESTABLISHED_PHARMACOLOGIC_CLASS/CELL/PRODUCT/TIME/HAZARDOUS_OR_POISONOUS_SUBSTANCE/HORMONE/DRUG_ADJECTIVE/MOUSE_PROTEIN_FAMILY/BIO/PHYSIOLOGIC_FUNCTION/CELL_FUNCTION/STUDY/SOCIAL_CIRCUMSTANCES/VIRAL_PROTEIN/CONGENITAL_ABNORMALITY/BIO_MOLECULE/BODY_SUBSTANCE/CELL_COMPONENT/BODY_LOCATION_OR_REGION/CHEMICAL_CLASS/ORGANISM_FUNCTION/BACTERIUM/MENTAL_OR_BEHAVIORAL_DYSFUNCTION/DEVICE/NUCLEOTIDE_SEQUENCE/VITAMIN/SPORT/CELL_OR_MOLECULAR_DYSFUNCTION/PRODUCT_ADJECTIVE/ORGANIZATION_ADJECTIVE/SEQUENCE/EDU 224/195/156/146/119/115/103/69/62/50/48/46/45/44/40/35/29/28/26/25/20/18/17/16/14/11/11/10/9/8/8/7/7/6/6/5/5/5/5/5/5/4/4/4/3/3/3/3/3/3/2/2/1/1/1/1/1/1 eGFR


**17 Sept 2021**

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

### Bootstrap labeling for NER file generartion.


When starting in a new domain with no labels, start with an empty labels.txt and bootstrap_entities.txt.  Cluster (run.sh with option 6). Add this to bootstrap_entities.txt list and repeat. 

When we start from scratch for a new domain we would need to label some terms as shown in this repo. A sample set is available for biomedical/phi/legal domain in  [NER](https://github.com/ajitrajasekharan/unsupervised_NER.git) 

# License

This repository is covered by MIT license. 
