import sys
from collections import OrderedDict
import pdb




def get_stats(file_name,index,stats_dict):
    with open(file_name,"r") as fp:
        labels_dict = {}
        line_count = 0
        for  line in fp:
            line = line.split()
            if (len(line) >= 2):
                if (line[0] == "_singletons_" or line[0] == "_empty_"):
                    continue
                label = line[index]
                labels_arr = label.rstrip('/').split('/')
                line_count += 1
                for curr_label in labels_arr:
                    if (len(curr_label) == 0):
                        print("check bootstrap file creation. Empty labels")
                        pdb.set_trace()
                    if (curr_label not in labels_dict):
                        labels_dict[curr_label]  = 1
                    else:
                        labels_dict[curr_label] += 1
        final_sorted = OrderedDict(sorted(labels_dict.items(), key=lambda kv: kv[1], reverse=True))
        singletons_count = 0
        for i in final_sorted:
            if (final_sorted[i] == 1):
                singletons_count += 1
        for i in final_sorted:
            if (final_sorted[i] != 1):
                print(i,final_sorted[i])
                if (i not in stats_dict):
                    stats_dict[i] = 1
        #print("SINGLETONS",singletons_count)



if __name__== "__main__":
    file_name = "adaptive_debug_pivots.txt"
    if (len(sys.argv) == 1):
        print("Assuming input file is ",file_name)
    else:
        file_name = sys.argv[1]
    stats_dict = {}
    get_stats(file_name,0,stats_dict)
    #print("SUB ENTITY STATS")
    #get_stats(file_name,1,stats_dict)
    with open("stats_dict.txt","w") as fp:
        for term in stats_dict:
            fp.write(term  + "\n")

