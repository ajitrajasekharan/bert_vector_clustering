import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string


import logging

#DEFAULT_MODEL_PATH='bert-large-cased'
DEFAULT_MODEL_PATH='./'
DEFAULT_TO_LOWER=False
DEFAULT_TOP_K = 20
ACCRUE_THRESHOLD = 1

def init_model(model_path,to_lower):
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    model = BertForMaskedLM.from_pretrained(model_path)
    #tokenizer = RobertaTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    #model = RobertaForMaskedLM.from_pretrained(model_path)
    model.eval()
    return model,tokenizer

def get_sent(to_lower):
    print("Enter sentence (optionally including the special token [MASK] if tolower is set to False). Type q to quit")
    sent = input()
    if (sent == 'q'):
        return sent
    else:
        return  sent.lower() if to_lower else sent

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n")
            if (len(line) >= 1):
                ret_dict[line] = 1
    return ret_dict



def perform_task(model,tokenizer,top_k,accrue_threshold,text):
    text = '[CLS]' + text + ' [SEP]' 
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    print(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])


    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
        for i in range(len(tokenized_text)):
                #if (i != 0 and i != len(tokenized_text) - 1):
                #    continue
                results_dict = {}
                masked_index = i
                neighs_dict = {}
                for j in range(len(predictions[0][0][0,masked_index])):
                    if (float(predictions[0][0][0,masked_index][j].tolist()) > accrue_threshold):
                        tok = tokenizer.convert_ids_to_tokens([j])[0]
                        results_dict[tok] = float(predictions[0][0][0,masked_index][j].tolist())
                    tok = tokenizer.convert_ids_to_tokens([j])[0]
                    if (tok in tokenized_text):
                        neighs_dict[tok] = float(predictions[0][0][0,masked_index][j].tolist())
                k = 0
                sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
                print("********* Top predictions for token: ",tokenized_text[i])
                for index in sorted_d:
                    if (index in string.punctuation or index.startswith('##') or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                        continue
                    print(index,round(float(sorted_d[index]),4))
                    k += 1
                    if (k > top_k):
                        break
                print("********* Closest sentence neighbors in output to the token :  ",tokenized_text[i])
                sorted_d = OrderedDict(sorted(neighs_dict.items(), key=lambda kv: kv[1], reverse=True))
                for index in sorted_d:
                    if (index in string.punctuation or index.startswith('##') or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                        continue
                    print(index,round(float(sorted_d[index]),4))
                print()
                print()
                pdb.set_trace()
                #break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting neighbors to a word in sentence using BERTMaskedLM. Neighbors are from BERT vocab (which includes subwords and full words). Type in a sentence and then choose a position to mask or type in a sentence with the word entity in the location to apply a mask ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-topk', action="store", dest="topk", default=DEFAULT_TOP_K,type=int,help='Number of neighbors to display')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)
    parser.add_argument('-threshold', action="store", dest="threshold", default=ACCRUE_THRESHOLD,type=float,help='threshold of results to pick')

    results = parser.parse_args()
    try:
        model,tokenizer = init_model(results.model,results.tolower)
        print("To lower casing is set to:",results.tolower)
        while (True):
            text = get_sent(results.tolower)
            if (text == "q"):
                print("Quitting")
                break
            perform_task(model,tokenizer,results.topk,results.threshold,text)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
