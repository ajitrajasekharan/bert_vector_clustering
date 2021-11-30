import torch
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string


DEFAULT_INPUT="adaptive_debug_pivots.txt"
DEFAULT_OUTPUT="confirm_weakly_tagged_terms.txt"

TAG_INDEX = 0
SCORE_INDEX = 1
REST_INDEX = 8


exclude_arr = ["DISEASE","DRUG","GENE","PROTEIN","OTHER"]

def perform_task(input_file,output_file):
    ofp = open(output_file,"w")
    with open(input_file) as ifp:
        for line in ifp:
            line = line.split()
            if (line[0] == "_empty_" or line[0] == "_singletons_"):
                continue
            entity = line[TAG_INDEX].strip("'")
            entity = entity.replace('/',' ').replace("UNTAGGED_ENTITY","").strip()
            entity = entity.replace(' ','/')
            entity = entity.split('/')
            if (len(entity) >= 1):
                entity = entity[0]
            else:
                continue
            if (entity in exclude_arr or len(entity) <= 1):
                continue
            score = line[SCORE_INDEX].strip("'")
            terms = ''.join(line[REST_INDEX:]).replace(' ','').split(',')
            print("XXXLINE:",line)
            ofp.write("XXXLINE:" + ' '.join(line) + "\n")
            for i in range(len(terms)):
                if (i == 0 ):
                    terms[i] = terms[i].lstrip('[')
                if (i == len(terms) -1):
                    terms[i] = terms[i].rstrip(']')
                terms[i] = terms[i].strip("'")
                if (terms[i].startswith("#")):
                    print(terms[i],entity)
                    ofp.write(terms[i] + " " + entity + "\n")
            ofp.flush()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weakly tag from clusters for human verification and selections',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input", default=DEFAULT_INPUT,help='BERT pretrained models, or custom model path')
    parser.add_argument('-output', action="store", dest="output", default=DEFAULT_OUTPUT,help='BERT pretrained models, or custom model path')

    results = parser.parse_args()
    try:
        perform_task(results.input,results.output)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
