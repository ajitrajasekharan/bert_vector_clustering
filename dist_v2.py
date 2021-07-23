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


SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG = "_empty_ "
OTHER_TAG = "OTHER"
AMBIGUOUS = "AMB"
MAX_VAL = 20
TAIL_THRESH = 10

BERT_TERMS_START=106
UNK_ID = 1
#Original setting for cluster generation. 
OLD_FORM=True

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict

def read_labels(labels_file):
    terms_dict = OrderedDict()
    with open(labels_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 5):
                terms_dict[term[2]] = {"label":term[0],"aux_label":term[1],"mean":float(term[3]),"variance":float(term[4])}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict


def read_entities(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                nodes = term.split()
                assert(len(nodes) == 2)
                terms_dict[nodes[1]] = nodes[0]
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

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
    if (OLD_FORM): 
        return True if (str(key).startswith('#') or str(key).startswith('[')) else False
    else:
        return True if (str(key).startswith('[')) else False

def filter_2g(term,preserve_dict):
    if (OLD_FORM):
        return True if  (len(term) <= 2 ) else False
    else:
        return True if  (len(term) <= 2 and term not in preserve_dict) else False

class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,cache_embeds,normalize,labels_file,stats_file,preserve_2g_file,glue_words_file,bootstrap_entities_file):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = read_terms(terms_file)
        self.labels_dict = read_labels(labels_file)
        self.stats_dict = read_terms(stats_file)
        self.preserve_dict = read_terms(preserve_2g_file)
        self.gw_dict = read_terms(glue_words_file)
        self.bootstrap_entities = read_entities(bootstrap_entities_file)
        self.embeddings = read_embeddings(embeds_file)
        self.cache = cache_embeds
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}
        self.dist_zero_cache = {}
        self.normalize = normalize

    def dump_vocab(self):
        #pdb.set_trace()
        size = self.tokenizer.vocab_size
        for i in range(size):
            names = self.tokenizer.convert_ids_to_tokens([i])
            print(names[0])


    def adaptive_gen_pivot_graphs(self):
        tokenize = False
        count = 1
        total = len(self.terms_dict)
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open("adaptive_debug_pivots.txt","w")
        for key in self.terms_dict:
            if (is_filtered_term(key) or count <= BERT_TERMS_START):
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
                entity_type = self.get_entity_type(arr)
                print(entity_type,new_key,max_mean,std_dev,arr)
                dfp.write(entity_type + " " + entity_type + " " + new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
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


    def get_entity_type(self,arr):
        e_dict = {} 
        for term in arr:
            if (term in self.bootstrap_entities):
                 if (self.bootstrap_entities[term] in e_dict):
                         e_dict[self.bootstrap_entities[term]] += 1
                 else:
                         e_dict[self.bootstrap_entities[term]] = 1
        
        if (len(e_dict) > 1):
               sorted_d = OrderedDict(sorted(e_dict.items(), key=lambda kv: kv[1], reverse=True))
               for k in sorted_d:
                   if (k != "OTHER" and sorted_d[k] >= 2):
                       return k
        return "OTHER"
      

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
            if (is_filtered_term(key) or count <= BERT_TERMS_START):
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
            if (is_filtered_term(key) or count <= BERT_TERMS_START):
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



    def get_embedding(self,text,tokenize=True):
        if (self.cache and text in self.embeds_cache):
            return self.embeds_cache[text]
        if (tokenize):
            tokenized_text = self.tokenizer.tokenize(text)
        else:
            if (not text.startswith('[')): 
               tokenized_text = text.split()
            else:
               tokenized_text = [text]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(text,indexed_tokens)
        vec =  self.get_vector(indexed_tokens)
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec

    def calc_inner_prod(self,text1,text2,tokenize):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1,tokenize)
        vec2 = self.get_embedding(text2,tokenize)
        if (vec1 is None or vec2 is None):
            #print("Warning: at least one of the vectors is None for terms",text1,text2)
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val

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

    def gen_label(self,node):
        if (node["label"] in self.stats_dict):
            if (node["aux_label"] in self.stats_dict):
                ret_label = node["label"] + "-" +  node["aux_label"]
            else:
                if (node["label"]  == AMBIGUOUS):
                    ret_label = node["label"] + "-" +  node["aux_label"]
                else:
                    ret_label = node["label"]
        else:
                ret_label = OTHER_TAG + "-" + node["label"]
        return ret_label

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
        tokenize = False
        words = self.filter_glue_words(words)
        desc_max_term,desc_mean,desc_std_dev,s_dict = self.find_pivot_subgraph(words,tokenize)
        print("PSG score of input descs:",desc_max_term,desc_mean,desc_std_dev,s_dict)
        #OVERRIDE pivot
        #desc_max_term = words[0]
        #print("pivot override",desc_max_term)
        pivot_similarities = {}
        for i,key in enumerate(entities):
            term = key
            val = round(self.calc_inner_prod(desc_max_term,term,tokenize),2)
            pivot_similarities[key] = val
            #print("SIM:",desc_max_term,term,val)
        sorted_d = OrderedDict(sorted(pivot_similarities.items(), key=lambda kv: kv[1], reverse=True))
        count = 0
        ret_arr = []
        for k in sorted_d:
            #if (sorted_d[k] < pick_threshold):
            #    if (count == 0):
            #        print("No entity above thresold")
            #    break
            print(entities[k]["label"],k,sorted_d[k],"(",entities[k]["mean"],entities[k]["variance"],")")
            ret_label = self.gen_label(entities[k])
            ret_arr.append(ret_label)
            count+= 1
            if (count >= 10):
                break
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
            print("Enter test type (0-gen cum dist for vocabs; 1-generate clusters (will take approx 2 hours);  2-neigh/3-pivot graph/4-bipartite/5-Entity test: q to quit")
            val = input()
            if (val == "0"):
                try:
                    b_embeds.gen_dist_for_vocabs()
                except:
                    print("Trapped exception")
                sys.exit(-1)
            elif (val == "1"):
                print("Enter Input threshold .5  works well for both pretraining and fine tuned. Enter 0 for adaptive thresholding")
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
            else:
                print("invalid option")




if __name__ == '__main__':
    main()
