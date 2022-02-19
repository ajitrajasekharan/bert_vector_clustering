import pdb
import operator
from collections import OrderedDict
import argparse
import sys
import traceback
import string



def read_labels_file(inp_file):
    terms_dict = OrderedDict()
    labels_dict = OrderedDict()
    labels_strength_dict = OrderedDict()
    total_count = 0
    total_strength_count = 0
    lc_vocab_dict = OrderedDict()
    with open(inp_file) as fp:
        for line in fp:
            line = line.rstrip("\n")
            line = line.split()
            labels = line[0].split("/")
            label_counts = line[1].split("/")
            term = line[2]
            if (term in terms_dict):
                pdb.set_trace()
            assert(term not in terms_dict)
            terms_dict[term] = labels
            lc_vocab_dict[term.lower()] = 1
            assert(len(labels) == len(label_counts))
            for label,count in zip(labels,label_counts):
                if (label not in labels_dict):
                    labels_strength_dict[label] = int(count)
                    labels_dict[label] = 1
                else:
                    labels_dict[label] += 1
                    labels_strength_dict[label] += int(count)
                total_count += 1
                total_strength_count += int(count)
    sorted_d = OrderedDict(sorted(labels_dict.items(), key=lambda kv: kv[1], reverse=True))
    return terms_dict,sorted_d,total_count,lc_vocab_dict,labels_strength_dict,total_strength_count


def read_bs_file(inp_file,vocab_names):
    terms_dict = OrderedDict()
    labels_dict = OrderedDict()
    total_count = 0
    with open(inp_file) as fp:
        for line in fp:
            line = line.rstrip("\n")
            line = line.split()
            labels = line[0].split("/")
            term = line[1]
            if (term not in vocab_names):
                continue
            if (term in terms_dict):
                pdb.set_trace()
            assert(term not in terms_dict)
            terms_dict[term] = labels
            for label in labels:
                if (label not in labels_dict):
                    labels_dict[label] = 1
                else:
                    labels_dict[label] += 1
                total_count += 1
    sorted_d = OrderedDict(sorted(labels_dict.items(), key=lambda kv: kv[1], reverse=True))
    return terms_dict,sorted_d,total_count

def get_counts(inp_dict):
    mag_terms_count = 0
    for key in inp_dict:
        if (len(inp_dict[key]) == 1 and (inp_dict[key][0] == "OTHER" or inp_dict[key][0] == "UNTAGGED_ENTITY")):
            continue
        mag_terms_count += 1
    return mag_terms_count

def find_predominant_entity(inp_dict):
    predom_dict = OrderedDict()
    for key in inp_dict:
        if (len(inp_dict[key]) == 1 and (inp_dict[key][0] == "OTHER" or inp_dict[key][0] == "UNTAGGED_ENTITY")):
            continue
        labels = inp_dict[key]
        for label in labels:
            if (label != "OTHER" and label != "UNTAGGED_ENTITY"):
                if (label not in predom_dict):
                    predom_dict[label] = 1
                else:
                    predom_dict[label] += 1
                break
    return predom_dict

def gen_mag_stats(params):
    bs_file = params.bootstrap_file
    labels_file = params.labels_file
    output_file = params.output_file
    mag_terms,mag_labels,label_total_count,lc_vocab_dict,labels_strength,total_strength_count = read_labels_file(labels_file)
    bs_terms, bs_labels,bs_total_count = read_bs_file(bs_file,lc_vocab_dict)
    orig_terms_count = get_counts(bs_terms)
    mag_terms_count = get_counts(mag_terms)
    predom_entity_dict = find_predominant_entity(mag_terms)
    fp = open(output_file,"w")
    text = "{}\t{}\t{}\t{}\t\t\t".format(len(mag_terms),mag_terms_count,orig_terms_count,str(round(float(mag_terms_count)/orig_terms_count,0)))
    print(text)
    fp.write(text + "\n")
    text = "{}\t{}\t{}\t{}\t\t\t".format(label_total_count,total_strength_count,bs_total_count,str(round(float(total_strength_count)/bs_total_count,0)))
    print(text)
    fp.write(text + "\n")
    for key in mag_labels:
        if (key in predom_entity_dict):
            predom_entity = predom_entity_dict[key]
        else:
            predom_entity = 0
        if (key in bs_labels):
            
            text = "{}\t{}\t{}\t{}\t{}\t{}".format(key,mag_labels[key],labels_strength[key],bs_labels[key],str(round(float(labels_strength[key])/bs_labels[key],0)),predom_entity)
        else:
            assert(key == "OTHER")
            text = "{}\t{}\t{}\t{}\t{}\t{}".format(key,mag_labels[key],labels_strength[key],0,labels_strength[key],predom_entity)
        if (key != "UNTAGGED_ENTITY" and key != "OTHER"):
            print(text)
            fp.write(text + "\n")
    fp.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Magnification stats ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-bootstrap_file', action="store", dest="bootstrap_file", default="bootstrap_entities.txt",help='bootstrap file with human and algo tagged labels')
    parser.add_argument('-labels_file', action="store", dest="labels_file", default="labels.txt",help='Labels magnified by clustering')
    parser.add_argument('-output_file', action="store", dest="output_file", default="mag_stats.tsv",help='Mag stats')
    results = parser.parse_args()
    try:
        gen_mag_stats(results) 
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
