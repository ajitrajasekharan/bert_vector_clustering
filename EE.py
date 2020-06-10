# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import dist_v2
import urllib.parse

singleton = None
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class EE(ResponseHandler.ResponseHandler):
	def handler(self,write_obj = None):
		print("In derived class")
		global singleton
		if singleton is None:
			singleton = dist_v2.BertEmbeds(os.getcwd(),0,'vocab.txt','bert_vectors.txt',True,True,'results/labels.txt','results/stats_dict.txt','preserve_1_2_grams.txt','glue_words.txt')
		if (write_obj is not None):
			param=urllib.parse.unquote(write_obj.path[1:])
			print("Arg = ",param)
			out = singleton.find_entities(param.split())
			out = ' '.join(out)
			print(out)
			if (len(out) >= 1):
				write_obj.wfile.write(str(out).encode())
			else:
				write_obj.wfile.write("0".encode())








def my_test():
    cl = EE()

    cl.handler()




#my_test()
