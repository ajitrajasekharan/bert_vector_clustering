import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import *
import sys
import random
import time


SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG = "_empty_ "
OTHER_TAG = "OTHER"
AMBIGUOUS = "AMB"
MAX_VAL = 20
TAIL_THRESH = 10
SUBWORD_COS_THRESHOLD = .1
MAX_SUBWORD_PICKS = 20

UNK_ID = 1
IGNORE_CONTINUATIONS=True
USE_PRESERVE=True

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_list = json.loads(fp.read())
    arr = np.array(embeds_list)
    return arr


def consolidate_labels(existing_node,new_labels,new_counts):
    """Consolidates all the labels and counts for terms ignoring casing

    For instance, egfr may not have an entity label associated with it
    but eGFR and EGFR may have. So if input is egfr, then this function ensures
    the combined entities set fo eGFR and EGFR is made so as to return that union
    for egfr
    """
    new_dict = {}
    existing_labels_arr = existing_node["label"].split('/')
    existing_counts_arr = existing_node["counts"].split('/')
    new_labels_arr = new_labels.split('/')
    new_counts_arr = new_counts.split('/')
    assert(len(existing_labels_arr) == len(existing_counts_arr))
    assert(len(new_labels_arr) == len(new_counts_arr))
    for i in range(len(existing_labels_arr)):
        new_dict[existing_labels_arr[i]] = int(existing_counts_arr[i])
    for i in range(len(new_labels_arr)):
        if (new_labels_arr[i] in new_dict):
            new_dict[new_labels_arr[i]] += int(new_counts_arr[i])
        else:
            new_dict[new_labels_arr[i]] = int(new_counts_arr[i])
    sorted_d = OrderedDict(sorted(new_dict.items(), key=lambda kv: kv[1], reverse=True))
    ret_labels_str = ""
    ret_counts_str = ""
    count = 0
    for key in sorted_d:
        if (count == 0):
            ret_labels_str = key
            ret_counts_str = str(sorted_d[key])
        else:
            ret_labels_str += '/' +  key
            ret_counts_str += '/' +  str(sorted_d[key])
        count += 1
    return {"label":ret_labels_str,"counts":ret_counts_str}




def read_labels(labels_file):
    terms_dict = OrderedDict()
    lc_terms_dict = OrderedDict()
    with open(labels_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 3):
                terms_dict[term[2]] = {"label":term[0],"counts":term[1]}
                lc_term = term[2].lower()
                if (lc_term in lc_terms_dict):
                     lc_terms_dict[lc_term] = consolidate_labels(lc_terms_dict[lc_term],term[0],term[1])
                else:
                     lc_terms_dict[lc_term] = {"label":term[0],"counts":term[1]}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict,lc_terms_dict


def read_entities(terms_file):
    ''' Read bootstrap entities file

    '''
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                nodes = term.split()
                assert(len(nodes) == 2)
                lc_node = nodes[1].lower()
                if (lc_node in terms_dict):
                    pdb.set_trace()
                    assert(0)
                    assert('/'.join(terms_dict[lc_node]) == nodes[0])
                terms_dict[lc_node] = nodes[0].split('/')
                count += 1
    print("count of entities in ",terms_file,":", len(terms_dict))
    return terms_dict



def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def is_subword(key):
        return True if str(key).startswith('#')  else False

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
    if (IGNORE_CONTINUATIONS):
        return True if (is_subword(key) or str(key).startswith('[')) else False
    else:
        return True if (str(key).startswith('[')) else False

def filter_2g(term,preserve_dict):
    if (USE_PRESERVE):
        return True if  (len(term) <= 2 and term not in preserve_dict) else False
    else:
        return True if  (len(term) <= 2 ) else False

class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,cache_embeds,normalize,labels_file,stats_file,preserve_2g_file,glue_words_file,bootstrap_entities_file):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = read_terms(terms_file)
        self.labels_dict,self.lc_labels_dict = read_labels(labels_file)
        self.stats_dict = read_terms(stats_file) #Not used anymore
        self.preserve_dict = read_terms(preserve_2g_file)
        self.gw_dict = read_terms(glue_words_file)
        self.bootstrap_entities = read_entities(bootstrap_entities_file)
        self.embeddings = read_embeddings(embeds_file)
        self.dist_threshold_cache = {}
        self.dist_zero_cache = {}
        self.normalize = normalize
        self.similarity_matrix = self.cache_matrix(True)

    def cache_matrix(self,normalize):
        b_embeds = self
        print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...)")
        start = time.time()
        #pdb.set_trace()
        vec_a = b_embeds.embeddings.T #vec_a shape (1024,)
        if (normalize):
            vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Norm is along axis 0 - rows
            vec_a = vec_a.T #vec_a shape becomes (,1024)
            similarity_matrix = np.inner(vec_a,vec_a)
        end = time.time()
        time_val = (end-start)*1000
        print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
        return similarity_matrix


    def dump_vocab(self):
        #pdb.set_trace()
        size = self.tokenizer.vocab_size
        for i in range(size):
            names = self.tokenizer.convert_ids_to_tokens([i])
            print(names[0])

    def labeled_term(self,k):
        if (k not in self.bootstrap_entities):
            return False
        labels = self.bootstrap_entities[k]
        if (len(labels) > 1):
            return True
        assert(len(labels) == 1)
        if (labels[0] == "UNTAGGED_ENTITY"):
            return False
        return True

    def subword_clustering(self):
        '''
               Generate clusters for terms in vocab
               This is used for unsupervised NER (with subword usage)
        '''
        tokenize = False
        count = 1
        total = len(self.terms_dict)
        pivots_dict = OrderedDict()
        singletons_arr = []
        full_entities_dict = OrderedDict()
        untagged_items_dict = OrderedDict()
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open("adaptive_debug_pivots.txt","w")
        esupfp = open("entity_support.txt","w")
        for key in self.terms_dict:
            if (key.startswith('[') or len(key) < 2):
                count += 1
                continue
            count += 1
            #print(":",key)
            print("Processing: ",key,"count:",count," of ",total)
            temp_sorted_d,dummy = self.get_distribution_for_term(key,False)
            sorted_d = self.get_terms_above_threshold(key,SUBWORD_COS_THRESHOLD,tokenize)
            arr = []
            for k in sorted_d:
                if (is_subword(k)):
                    continue
                if (not self.labeled_term(k.lower())):
                    continue
                arr.append(k)
                if (len(arr) > MAX_SUBWORD_PICKS):
                    break
            if (len(arr) > MAX_SUBWORD_PICKS/2):
                max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr,tokenize)
                if (max_mean_term not in pivots_dict):
                    new_key  = max_mean_term
                else:
                    print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                    new_key  = max_mean_term + "++" + key
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                entity_type,entity_counts,curr_entities_dict = self.get_entity_type(arr,new_key,esupfp)
                self.aggregate_entities_for_terms(arr,curr_entities_dict,full_entities_dict,untagged_items_dict)
                print(entity_type,entity_counts,new_key,max_mean,std_dev,arr)
                dfp.write(entity_type + " " + entity_counts + " " + new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
            else:
                if (len(arr) != 0):
                    print("***Sparse arr for term:",key)
                    singletons_arr.append(key)
                else:
                    print("***Empty arr for term:",key)
                    empty_arr.append(key)
            #if (count >= 500):
            #    break

        dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
        dfp.write(EMPTY_TAG + str(empty_arr) + "\n")
        with open("pivots.json","w") as fp:
            fp.write(json.dumps(pivots_dict))
        with open("pivots.txt","w") as fp:
            for k in pivots_dict:
                fp.write(k + '\n')
        dfp.close()
        esupfp.close()
        self.create_entity_labels_file(full_entities_dict)
        self.create_inferred_entities_file(untagged_items_dict)


    def adaptive_gen_pivot_graphs(self):
        '''
               Generate clusters for terms in vocab
               This is used for unsupervised NER
        '''
        tokenize = False
        count = 1
        total = len(self.terms_dict)
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        full_entities_dict = OrderedDict()
        untagged_items_dict = OrderedDict()
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open("adaptive_debug_pivots.txt","w")
        esupfp = open("entity_support.txt","w")
        for key in self.terms_dict:
            if (is_filtered_term(key)):
                count += 1
                continue
            count += 1
            #print(":",key)
            if (key in picked_dict or len(key) <= 2):
                continue
            print("Processing ",count," of ",total)
            picked_dict[key] = 1
            temp_sorted_d,dummy = self.get_distribution_for_term(key,False)
            dummy,threshold = self.get_tail_length(key,temp_sorted_d)
            sorted_d = self.get_terms_above_threshold(key,threshold,tokenize)
            arr = []
            for k in sorted_d:
                if (is_filtered_term(k) or filter_2g(k,self.preserve_dict)):
                    picked_dict[k] = 1
                    continue
                picked_dict[k] = 1
                arr.append(k)
            if (len(arr) > 1):
                max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr,tokenize)
                if (max_mean_term not in pivots_dict):
                    new_key  = max_mean_term
                else:
                    print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                    new_key  = max_mean_term + "++" + key
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                entity_type,entity_counts,curr_entities_dict = self.get_entity_type(arr,new_key,esupfp)
                self.aggregate_entities_for_terms(arr,curr_entities_dict,full_entities_dict,untagged_items_dict)
                print(entity_type,entity_counts,new_key,max_mean,std_dev,arr)
                dfp.write(entity_type + " " + entity_counts + " " + new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
            else:
                if (len(arr) == 1):
                    print("***Singleton arr for term:",key)
                    singletons_arr.append(key)
                else:
                    print("***Empty arr for term:",key)
                    empty_arr.append(key)
            #if (count >= 500):
            #    break

        dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
        dfp.write(EMPTY_TAG + str(empty_arr) + "\n")
        with open("pivots.json","w") as fp:
            fp.write(json.dumps(pivots_dict))
        with open("pivots.txt","w") as fp:
            for k in pivots_dict:
                fp.write(k + '\n')
        dfp.close()
        esupfp.close()
        self.create_entity_labels_file(full_entities_dict)
        self.create_inferred_entities_file(untagged_items_dict)

    def aggregate_entities_for_terms(self,arr,curr_entities_dict,full_entities_dict,untagged_items_dict):
        if (len(curr_entities_dict) == 0):
            return
        for term in arr:
            if (term.lower() in self.bootstrap_entities): #Note this is a case insensitive check
                term_entities = self.bootstrap_entities[term.lower()]
            else:
                if (term not in untagged_items_dict):
                    untagged_items_dict[term] = OrderedDict()
                for entity in curr_entities_dict:
                    if (entity not in untagged_items_dict[term]):
                        untagged_items_dict[term][entity] = curr_entities_dict[entity]
                    else:
                        untagged_items_dict[term][entity] += curr_entities_dict[entity]
                continue
            #We come here only for terms that were present in the bootstrap list
            if term not in full_entities_dict: #This is case sensitive. We want vocab entries eGFR and EGFR to pick up separate weights for their entities
                full_entities_dict[term] = OrderedDict()
            for entity in curr_entities_dict:
                if  (entity not in term_entities): #aggregate counts only for entities present for this term in original manual harvesting list(bootstrap list)
                    continue
                if (entity not  in full_entities_dict[term]):
                    full_entities_dict[term][entity] = curr_entities_dict[entity]
                else:
                    full_entities_dict[term][entity] += curr_entities_dict[entity]


    def create_entity_labels_file(self,full_entities_dict):
        with open("labels.txt","w") as fp:
            for term in self.terms_dict:
                if (term not in full_entities_dict and term.lower() not in self.bootstrap_entities):
                    fp.write("OTHER 0 " + term + "\n")
                    continue
                if (term not in full_entities_dict): #These are vocab terms that did not show up in a cluster but are present in bootstrap list
                    lc_term = term.lower()
                    counts_str = len(self.bootstrap_entities[lc_term])*"0/"
                    fp.write('/'.join(self.bootstrap_entities[lc_term]) + ' ' + counts_str.rstrip('/') + ' ' + term + '\n') #Note the term output is case sensitive. Just the indexed version is case insenstive
                    continue
                out_entity_dict = {}
                for entity in full_entities_dict[term]:
                    assert(entity not in out_entity_dict)
                    out_entity_dict[entity] = full_entities_dict[term][entity]
                sorted_d = OrderedDict(sorted(out_entity_dict.items(), key=lambda kv: kv[1], reverse=True))
                entity_str = ""
                count_str = ""
                for entity in sorted_d:
                    if (len(entity_str) == 0):
                        entity_str = entity
                        count_str =  str(sorted_d[entity])
                    else:
                        entity_str += '/' +  entity
                        count_str +=  '/' + str(sorted_d[entity])
                if (len(entity_str) > 0):
                    fp.write(entity_str + ' ' + count_str + ' ' + term + "\n")


    def sort_and_consolidate_inferred_entities_file(self,untagged_items_dict):
            for term in untagged_items_dict:
                out_entity_dict = {}
                for entity in untagged_items_dict[term]:
                    assert(entity not in out_entity_dict)
                    out_entity_dict[entity] = untagged_items_dict[term][entity]
                sorted_d = OrderedDict(sorted(out_entity_dict.items(), key=lambda kv: kv[1], reverse=True))
                first = next(iter(sorted_d))
                #untagged_items_dict[term] = {first:sorted_d[first]} #Just pick the first entity
                untagged_items_dict[term] = sorted_d


            ci_untagged_items_dict = OrderedDict()
            for term in untagged_items_dict:
                lc_term = term.lower()
                if (lc_term not in ci_untagged_items_dict):
                    ci_untagged_items_dict[lc_term] =  OrderedDict()
                for entity in untagged_items_dict[term]:
                    if (entity not in ci_untagged_items_dict[lc_term]):
                        ci_untagged_items_dict[lc_term][entity] = untagged_items_dict[term][entity]
                    else:
                        ci_untagged_items_dict[lc_term][entity] += untagged_items_dict[term][entity]
            return ci_untagged_items_dict


    def create_inferred_entities_file(self,untagged_items_dict):
        with open("inferred.txt","w") as fp:
            untagged_items_dict = self.sort_and_consolidate_inferred_entities_file(untagged_items_dict)
            for term in untagged_items_dict:
                out_entity_dict = {}
                for entity in untagged_items_dict[term]:
                    assert(entity not in out_entity_dict)
                    out_entity_dict[entity] = untagged_items_dict[term][entity]
                sorted_d = OrderedDict(sorted(out_entity_dict.items(), key=lambda kv: kv[1], reverse=True))
                entity_str = ""
                count_str = ""
                count_val = 0
                for entity in sorted_d:
                    if (len(entity_str) == 0):
                        entity_str = entity
                        count_str =  str(sorted_d[entity])
                    else:
                        entity_str += '/' +  entity
                        count_str +=  '/' + str(sorted_d[entity])
                    count_val += int(sorted_d[entity])
                if (len(entity_str) > 0):
                    fp.write(entity_str + ' ' + count_str + ' ' + str(count_val) + ' ' + term + "\n")


    def get_entity_type(self,arr,new_key,esupfp):
        e_dict = {}
        #print("GET:",arr)
        for term in arr:
            term = term.lower() #bootstrap entities is all lowercase.
            if (term in self.bootstrap_entities):
                 entities = self.bootstrap_entities[term]
                 for entity in entities:
                       if (entity in e_dict):
                            #print(term,entity)
                            e_dict[entity] += 1
                       else:
                            #print(term,entity)
                            e_dict[entity] = 1
        ret_str = ""
        count_str = ""
        entities_dict = OrderedDict()
        if (len(e_dict) >= 1):
               sorted_d = OrderedDict(sorted(e_dict.items(), key=lambda kv: kv[1], reverse=True))
               #print(new_key + ":" + str(sorted_d))
               esupfp.write(new_key + ' ' + str(sorted_d) + '\n')
               count = 0
               for k in sorted_d:
                   if (len(ret_str) > 0):
                       ret_str += '/' + k
                       count_str += '/' + str(sorted_d[k])
                   else:
                       ret_str = k
                       count_str = str(sorted_d[k])
                   entities_dict[k] = int(sorted_d[k])
                   count += 1
        if (len(ret_str) <= 0):
            ret_str = "OTHER"
            count_str = str(len(arr))
        #print(ret_str)
        count_str += '/' + str(len(arr))
        return ret_str,count_str,entities_dict


    def fixed_gen_pivot_graphs(self,threshold,count_limit):
        tokenize = False
        count = 1
        total = len(self.terms_dict)
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open("debug_pivots.txt","w")
        for key in self.terms_dict:
            if (is_filtered_term(key) ):
                count += 1
                continue
            count += 1
            #print(":",key)
            if (key in picked_dict or len(key) <= 2):
                continue
            print("Processing ",count," of ",total)
            picked_dict[key] = 1
            sorted_d = self.get_terms_above_threshold(key,threshold,tokenize)
            arr = []
            for k in sorted_d:
                if (is_filtered_term(k) or filter_2g(k,self.preserve_dict)):
                    picked_dict[k] = 1
                    continue
                if (sorted_d[k] < count_limit):
                    picked_dict[k] = 1
                    arr.append(k)
                else:
                    break
            if (len(arr) > 1):
                max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr,tokenize)
                if (max_mean_term not in pivots_dict):
                    new_key  = max_mean_term
                else:
                    print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                    new_key  = max_mean_term + "++" + key
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                print(new_key,max_mean,std_dev,arr)
                dfp.write(new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
            else:
                if (len(arr) == 1):
                    print("***Singleton arr for term:",key)
                    singletons_arr.append(key)
                else:
                    print("***Empty arr for term:",key)
                    empty_arr.append(key)

        dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
        dfp.write(EMPTY_TAG + str(empty_arr) + "\n")
        with open("pivots.json","w") as fp:
            fp.write(json.dumps(pivots_dict))
        dfp.close()


    def get_tail_length(self,key,sorted_d):
        rev_sorted_d = OrderedDict(sorted(sorted_d.items(), key=lambda kv: kv[0], reverse=True))
        prev_val = 0
        prev_cosine_val = 0
        count = 0
        cosine_val = 0
        for k in rev_sorted_d:
            if (rev_sorted_d[k] >= MAX_VAL):
                   if (prev_val >= TAIL_THRESH):
                        count -= prev_val
                        cosine_val = prev_cosine_val
                   else:
                        cosine_val = k
                   break
            if (rev_sorted_d[k] >= TAIL_THRESH and prev_val >= TAIL_THRESH):
                   count -= prev_val
                   cosine_val = prev_cosine_val
                   break
            prev_val = rev_sorted_d[k]
            prev_cosine_val = k
            count += rev_sorted_d[k]
        return count,cosine_val




    def gen_dist_for_vocabs(self):
        print("Random pick? (Full run will take approximately 3 hours) Y/n:")
        resp = input()
        is_rand = (resp == "Y")
        if (is_rand):
            print("Sampling run:")
        count = 1
        picked_count = 0
        skip_count = 0
        cum_dict = OrderedDict()
        cum_dict_count = OrderedDict()
        zero_dict = OrderedDict()
        tail_lengths = OrderedDict()
        total_tail_length = 0
        for key in self.terms_dict:
            if (is_filtered_term(key) ):
                count += 1
                continue
            if (is_rand):
                val = random.randint(0,100)
                if (val < 97): # this is a biased skip to do a fast cum dist check (3% sample ~ 1000)
                    skip_count+= 1
                    print("Processed:",picked_count,"Skipped:",skip_count,end='\r')
                    continue
            #print(":",key)
            picked_count += 1
            sorted_d,dummy = self.get_distribution_for_term(key,False)
            tail_len,dummy = self.get_tail_length(key,sorted_d)
            tail_lengths[key] = tail_len
            total_tail_length += tail_len
            for k in sorted_d:
                val = round(float(k),1)
                #print(str(val)+","+str(sorted_d[k]))
                if (val == 0):
                    zero_dict[key] = sorted_d[k]
                if (val in cum_dict):
                    cum_dict[val] += sorted_d[k]
                    cum_dict_count[val] += 1
                else:
                    cum_dict[val] = sorted_d[k]
                    cum_dict_count[val] = 1
        for k in cum_dict:
            cum_dict[k] = round(float(cum_dict[k])/cum_dict_count[k],0)
        final_sorted_d = OrderedDict(sorted(cum_dict.items(), key=lambda kv: kv[0], reverse=False))
        print("\nTotal picked:",picked_count)
        with open("cum_dist.txt","w") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            for k in final_sorted_d:
                print(k,final_sorted_d[k])
                p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                fp.write(p_str)

        with open("zero_vec_counts.txt","w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            final_sorted_d = OrderedDict(sorted(zero_dict.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except:
                print("Exception 1")

        with open("tail_counts.txt","w",encoding="utf-8") as fp:
            fp.write("Total picked:" + str(picked_count) + " Average tail len: " + str(round(float(total_tail_length)/picked_count,1)) +  "\n")
            final_sorted_d = OrderedDict(sorted(tail_lengths.items(), key=lambda kv: kv[1], reverse=True))
            try:
                for k in final_sorted_d:
                    #print(k,final_sorted_d[k])
                    p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                    fp.write(p_str)
            except:
                print("Exception 2")



    def get_embedding_index(self,text,tokenize=False):
        if (tokenize):
            assert(0)
            tokenized_text = self.tokenizer.tokenize(text)
        else:
            if (not text.startswith('[')):
               tokenized_text = text.split()
            else:
               tokenized_text = [text]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        assert(len(indexed_tokens) == 1)
        return indexed_tokens[0]



    def calc_inner_prod(self,text1,text2,tokenize):
        assert(tokenize == False)
        index1 = self.get_embedding_index(text1)
        index2 = self.get_embedding_index(text2)
        return self.similarity_matrix[index1][index2]

    def get_distribution_for_term(self,term1,tokenize):
        debug_fp = None
        hack_check = False

        if (term1 in self.dist_threshold_cache):
            return self.dist_threshold_cache[term1],self.dist_zero_cache
        terms_count = self.terms_dict
        dist_dict = {}
        val_dict = {}
        zero_dict = {}
        if (hack_check and debug_fp is None):
            debug_fp = open("debug.txt","w")
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2,tokenize)
            #if (hack_check and val >= .8 and term1 != term2):
            if (hack_check and val >= .6 and val < .8 and term1 != term2):
                print(term1,term2)
                str_val = term1 + " " + term2 + "\n"
                debug_fp.write(str_val)
                debug_fp.flush()

            val = round(val,2)
            if (val in dist_dict):
                dist_dict[val] += 1
            else:
                dist_dict[val] = 1
            val = round(val,1)
            if (val >= -.05 and val <= .05):
                zero_dict[term2] = 0
        sorted_d = OrderedDict(sorted(dist_dict.items(), key=lambda kv: kv[0], reverse=False))
        self.dist_threshold_cache[term1] = sorted_d
        self.dist_zero_cache = zero_dict
        return sorted_d,zero_dict

    def get_terms_above_threshold(self,term1,threshold,tokenize):
        final_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2,tokenize)
            val = round(val,2)
            if (val > threshold):
                final_dict[term2] = val
        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
        return sorted_d

    def print_terms_above_threshold(self,term1,threshold,tokenize):
        fp = open("above_t.txt","w")
        sorted_d = self.get_terms_above_threshold(term1,threshold,tokenize)
        for k in sorted_d:
                print(k," ",sorted_d[k])
                fp.write(str(k) + " " + str(sorted_d[k]) + "\n")
        fp.close()


    #given n terms, find the mean of the connection strengths of subgraphs considering each term as pivot.
    #return the mean of max strength term subgraph
    def find_pivot_subgraph(self,terms,tokenize):
        max_mean = 0
        std_dev = 0
        max_mean_term = None
        means_dict = {}
        if (len(terms) == 1):
            return terms[0],1,0,{terms[0]:1}
        for i in terms:
            full_score = 0
            count = 0
            full_dict = {}
            for j in terms:
                if (i != j):
                    val = self.calc_inner_prod(i,j,tokenize)
                    #print(i+"-"+j,val)
                    full_score += val
                    full_dict[count] = val
                    count += 1
            if (len(full_dict) > 0):
                mean  =  float(full_score)/len(full_dict)
                means_dict[i] = mean
                #print(i,mean)
                if (mean > max_mean):
                    #print("MAX MEAN:",i)
                    max_mean_term = i
                    max_mean = mean
                    std_dev = 0
                    for k in full_dict:
                        std_dev +=  (full_dict[k] - mean)*(full_dict[k] - mean)
                    std_dev = math.sqrt(std_dev/len(full_dict))
                    #print("MEAN:",i,mean,std_dev)
        #print("MAX MEAN TERM:",max_mean_term)
        sorted_d = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))
        return max_mean_term,round(max_mean,2),round(std_dev,2),sorted_d




    def calc_bipartite_graph_strength_score(self,terms1,terms2,tokenize,normalize):
        full_score = 0
        max_val = 0
        for i in terms1:
            for j in terms2:
                    val = self.calc_inner_prod(i,j,tokenize)
                    print(i,j,val)
                    if (val > max_val):
                        max_val = val
                    full_score += val
        val = float(full_score)/(len(terms1)*len(terms2)) if normalize else float(full_score)
        return round(val,2),round(max_val,2)


    def filter_glue_words(self,words):
        ret_words = []
        for dummy,i in enumerate(words):
            if (i not in self.gw_dict):
                ret_words.append(i)
        if (len(ret_words) == 0):
            ret_words.append(words[0])
        return ret_words

    def find_entities(self,words):
        entities = self.labels_dict
        lc_entities = self.lc_labels_dict
        #words = self.filter_glue_words(words) #do not filter glue words anymore. Let them pass through
        ret_arr = []
        for word in words:
            l_word = word.lower()
            if l_word.isdigit():
                ret_label = "MEASURE"
                ret_counts = str(1)
            elif (word in entities):
                ret_label = entities[word]["label"]
                ret_counts = entities[word]["counts"]
            elif (l_word in entities):
                ret_label = entities[l_word]["label"]
                ret_counts = entities[l_word]["counts"]
            elif (l_word in lc_entities):
                ret_label = lc_entities[l_word]["label"]
                ret_counts = lc_entities[l_word]["counts"]
            else:
                ret_label = "OTHER"
                ret_counts = "1"
            if (ret_label == "OTHER"):
                ret_label = "UNTAGGED_ENTITY"
                ret_counts = "1"
            print(word,ret_label,ret_counts)
            ret_arr.append(ret_label)
            ret_arr.append(ret_counts)
        return ret_arr




def get_word():
    while (True):
        print("Enter a word : q to quit")
        sent = input()
        #print(sent)
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        if (len(sent) > 0):
            break
    return sent


def get_words():
    while (True):
        print("Enter words separated by spaces : q to quit")
        sent = input()
        #print(sent)
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        if (len(sent) > 0):
            break
    return sent.split()



def pick_threshold():
    while (True):
        print("Enter threshold to see words above threshold: q to quit")
        sent = input()
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        try:
            thres = float(sent)
            return thres
        except:
            print("Invalid input. Retry")


def neigh_test(b_embeds,tokenize):
    while (True):
        word = get_word()
        if (tokenize):
            tokenized_text = b_embeds.tokenizer.tokenize(word)
            print("Tokenized text:", tokenized_text)
        sorted_d,zero_dict = b_embeds.get_distribution_for_term(word,tokenize)
        for k in sorted_d:
            print(str(k)+","+str(sorted_d[k]))
        if (tokenize):
            print("Tokenized text:", tokenized_text)
        else:
            indexed_tokens = b_embeds.tokenizer.convert_tokens_to_ids(word)
            if (indexed_tokens == UNK_ID):
                print("Warning! This is not a token in vocab. Distribution is for UNK token")
        fp = open(word +"_zero.txt","w")
        for term in zero_dict:
            fp.write(term + '\n')
        fp.close()
        threshold = pick_threshold()
        b_embeds.print_terms_above_threshold(word,threshold,tokenize)

def graph_test(b_embeds,tokenize):
    while (True):
        words = get_words()
        max_mean_term,max_mean, std_dev,s_dict = b_embeds.find_pivot_subgraph(words,True)
        desc = ""
        for i in s_dict:
            desc += i + " "
        print("PSG score:",max_mean_term,max_mean, std_dev,s_dict)
        print(desc)

def bipartite_test(b_embeds,tokenize):
    while (True):
        print("First set")
        words1 = get_words()
        print("Second set")
        words2 = get_words()
        print("BF score:",b_embeds.calc_bipartite_graph_strength_score(words1,words2,True,False))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)




def impl_entities(b_embeds,tokenize,pick_threshold):
    while (True):
        words = get_words()
        ret_arr = b_embeds.find_entities(words)
        print(' '.join(ret_arr))


def main():
    if (len(sys.argv) != 11):
        print("Usage: <Bert model path - to load tokenizer> do_lower_case[1/0] <vocab file> <vector file> <tokenize text>1/0 <labels_file>  <preserve_1_2_grams_file> < glue words file> <bootstrap entities file>")
    else:
        tokenize = True if int(sys.argv[5]) == 1 else False
        if (tokenize == True):
            print("Forcing tokenize to false. Ignoring input value")
            tokenize = False #Adding this override to avoid inadvertant subword token generation error for pivot cluster generation
        print("Tokenize is set to :",tokenize)
        b_embeds =BertEmbeds(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],True,True,sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10]) #True - for cache embeds; normalize - True
        display_threshold = .4
        while (True):
            print("Enter test type (0-gen cum dist for vocabs; 1-generate clusters (will take approx 2 hours);  2-neigh/3-pivot graph/4-bipartite/5-Entity test/6-Subword neighbor cluster: q to quit")
            val = input()
            if (val == "0"):
                try:
                    b_embeds.gen_dist_for_vocabs()
                except:
                    print("Trapped exception")
                sys.exit(-1)
            elif (val == "1"):
                print("Enter Input threshold .5  works well for both pretraining and fine tuned. Enter 0 for adaptive thresholding(0 is recommended)")
                val = .5
                tail = 10
                try:
                    val = float(input())
                except:
                    val = .5
                if (val != 0):
                        print("Using value for fixed thresholding: ",val)
                        b_embeds.fixed_gen_pivot_graphs(val,tail)
                else:
                        print("Performing adaptive thresholding")
                        b_embeds.adaptive_gen_pivot_graphs()
                sys.exit(-1)
            elif (val == "2"):
                neigh_test(b_embeds,tokenize)
            elif (val == "3"):
                graph_test(b_embeds,tokenize)
            elif (val == "4"):
                bipartite_test(b_embeds,tokenize)
            elif (val == "5"):
                impl_entities(b_embeds,tokenize,display_threshold)
            elif (val == 'q'):
                sys.exit(-1)
            elif (val == "6"):
                 b_embeds.subword_clustering()
            else:
                print("invalid option")




if __name__ == '__main__':
    main()
