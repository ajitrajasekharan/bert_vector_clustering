import torch
import pdb
import sys


#dump_key = "cls.predictions.decoder.weight"
default_dump_key= "bert.embeddings.word_embeddings.weight"
default_model_name = "./pytorch_model.bin"
Usage = "Usage:  examine_model <1/2> \n\t\t1-for dumping pytorch model (python dict).\n\t\t2- for dumping word vectors\n\t<Specify model file name(optional)> - default pytorch_model.bin\n\t<Specify key name to dump word vectors(optional - used with option 2)>\n"


def examine(command,model_name,key_name):
	md = torch.load(model_name,map_location='cpu')
	if (command == 1):
		for k in md:
		    print(k)
	elif (command == 2):
		for k in md:
			if (k == key_name):
				embeds = md[k]
				print(embeds.tolist())
	else:
		print("Invalid command option:\n" + Usage)



if __name__ == "__main__":
	if (len(sys.argv) < 2):
		print(Usage)
	else:
		if (len(sys.argv) > 2):
			model_name = sys.argv[2]	
		else:
			model_name = default_model_name
		if (len(sys.argv) > 3):
			key_name = sys.argv[2]	
		else:
			key_name = default_dump_key
		examine(int(sys.argv[1]),model_name,key_name)
