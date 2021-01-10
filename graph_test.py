import sys
import json
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pdb
import numpy as  np
from collections import OrderedDict

def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict


class CL_SE:
    def __init__(self, model_path,vector_file,vocab_file,is_mlm):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        if (is_mlm):
            self.model = BertForMaskedLM.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained(model_path)
        self.embeddings = read_embeddings(vector_file)
        self.terms_dict = read_terms(vocab_file)


    def output_vectors_test(self,sent,is_patched):
        text = '[CLS]' + sent + '[SEP]' 
        tokenized_text = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(sent, return_tensors="pt")
        outputs = self.model(**inputs)
        count = 0
        print(tokenized_text,len(inputs['input_ids'][0]))
        assert(len(tokenized_text) == len(inputs['input_ids'][0]))
        #if (is_patched):
        #        print(len(outputs[0][1][0]))
        #else:
        #        print(len(outputs[0][0]))
        #print(len(tokenized_text))
        tot_len = len(tokenized_text)
        i = 0
        arr = outputs[0][1][0] if is_patched else outputs[0][0]
        for vec_i in arr:
            j = 0
            scores_dict = {}
            v1 = vec_i.tolist()
            tok1 = str(i) + '_' + tokenized_text[i]
            n1 =  np.linalg.norm(v1)
            for term in  self.terms_dict:
                indexed_tokens = self.tokenizer.convert_tokens_to_ids([term])
                v2 = self.embeddings[indexed_tokens[0]]
                bias = outputs[0][2][j].tolist() if is_patched else 0
                val = np.inner(v1,v2) + bias
                n2 = np.linalg.norm(v2)
                n_val = np.inner(v1/n1,v2/n2)
                tok2 = term
                scores_dict[tok1 + '_' + tok2] = {"val":val,"cos":n_val,"n1":n1,"n2":n2,"bias":bias}
                j += 1
            i += 1
            final_sorted_d = OrderedDict(sorted(scores_dict.items(), key=lambda kv: kv[1]["val"], reverse=True))
            count = 0
            for term in final_sorted_d:
                print(term,final_sorted_d[term])
                count += 1
                if (count >= 40):
                    break
            pdb.set_trace()
            #break
            print()
            print()
    #END OUTPUT VECTORS



def main(model_path,vector_file,vocab_file,p_is_patched,p_is_mlm):
    is_patched = True if p_is_patched == 1 else False
    is_mlm = True if p_is_mlm ==  1 else False
    se = CL_SE(model_path,vector_file,vocab_file,is_mlm)
    #se.output_vectors_test("epidermal growth factor receptor",is_patched)
    while (True):
        print("Enter sentence optionally including [MASK] token with proper casing: ")
        sent = input()
        se.output_vectors_test(sent,is_patched)

            
            


if __name__ == "__main__":
    if (len(sys.argv) != 6):
        print("Usage prog <model path> <vector file extracted from model> <vocab file> <is patched 1/0> <is MLM head vectors 1/0>")
    else: 
            main(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
