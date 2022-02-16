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
    pivot_labels_dict = OrderedDict()
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
            assert(len(labels) == len(label_counts))
            first = True
            for label,count in zip(labels,label_counts):
                if (first):
                    first = False
                    if (label not in pivot_labels_dict):
                        pivot_labels_dict[label] = 1
                    else:
                        pivot_labels_dict[label] += 1
                if (label not in labels_dict):
                    labels_dict[label] = int(count)
                else:
                    labels_dict[label] += int(count)
    sorted_d = OrderedDict(sorted(labels_dict.items(), key=lambda kv: kv[1], reverse=True))
    pivot_sorted_d = OrderedDict(sorted(pivot_labels_dict.items(), key=lambda kv: kv[1], reverse=True))
    return terms_dict,sorted_d,pivot_sorted_d


def read_bs_file(inp_file):
    terms_dict = OrderedDict()
    labels_dict = OrderedDict()
    with open(inp_file) as fp:
        for line in fp:
            line = line.rstrip("\n")
            line = line.split()
            labels = line[0].split("/")
            term = line[1]
            if (term in terms_dict):
                pdb.set_trace()
            assert(term not in terms_dict)
            terms_dict[term] = labels
            for label in labels:
                if (label not in labels_dict):
                    labels_dict[label] = 1
                else:
                    labels_dict[label] += 1
    sorted_d = OrderedDict(sorted(labels_dict.items(), key=lambda kv: kv[1], reverse=True))
    return terms_dict,sorted_d

def get_counts(inp_dict):
    mag_terms_count = 0
    for key in inp_dict:
        if (len(inp_dict[key]) == 1 and (inp_dict[key][0] == "OTHER" or inp_dict[key][0] == "UNTAGGED_ENTITY")):
            continue
        mag_terms_count += 1
    return mag_terms_count


def gen_mag_stats(params):
    bs_file = params.bootstrap_file
    labels_file = params.labels_file
    output_file = params.output_file
    bs_terms, bs_labels = read_bs_file(bs_file)
    mag_terms,mag_labels,pivot_mag_labels = read_labels_file(labels_file)
    orig_terms_count = get_counts(bs_terms)
    mag_terms_count = get_counts(mag_terms)
    fp = open(output_file,"w")
    text = "{}\t{}\t{}".format(str(round(float(mag_terms_count)/orig_terms_count,2)),mag_terms_count,orig_terms_count)
    print(text)
    fp.write(text + "\n")
    for key in mag_labels:
        if (key in bs_labels):
            mag_key_count = pivot_mag_labels[key] if key in pivot_mag_labels else 0
            text = "{}\t{}\t{}\t{}".format(key,bs_labels[key],mag_labels[key],mag_key_count)
        else:
            assert(key == "OTHER")
            text = "{}\t{}\t{}\t{}".format(key,0,mag_labels[key],pivot_mag_labels[key])
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
