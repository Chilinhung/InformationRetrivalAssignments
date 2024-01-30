# BM25 + BM25 Rocchio + Similarity
##################################

import os
import numpy as np
from numpy import zeros
import math
from math import log
from pylab import random
from time import perf_counter
import copy
import json
import nltk
from pathlib import Path
#word
word_set = set()
word_dict = {}
word_dict_r = {}
idf_dict = {}

#doc
no_doc = {}
doc_no = {}
docs_tf_list = []
docs_len_dict = {}
docs_count = 20000
docs_totalword = 0
avg_docs_len = 0

#query
no_qry = {}
qrys_tf_list = []
qry_related_5doc_list = []
qrys_len_dict = {}
qry_count = 100

def read_doc(path):
	global docs_tf_list
	global unigram_list
	global word_set
	global docs_len_dict
	global no_doc
	global docs_totalword
	global word_bg_dict
	os.chdir(path)
	file_names = os.listdir()
	dc = 0
	for file in os.listdir():
		if file.endswith(".txt"):
			file_path = f"{path}/{file}"
			with open(file_path, 'r') as f:
				doc_wd = []
				docj_tf_dict = {}
				for line in f:
					for w in line.split():
						if w.isnumeric(): #True in [char.isdigit() for char in w]:
							continue
						doc_wd.append(w)
						if w in docj_tf_dict:
							docj_tf_dict[w] += 1
						else:
							docj_tf_dict[w] = 1
						word_set.add(w)
				len_doc = len(doc_wd)
				docs_len_dict[file_names[dc]] = len_doc #doc j wordcount
				docs_totalword += len_doc
				dc += 1
			docs_tf_list.append(docj_tf_dict)		
	num = 0
	for dc in file_names: #create no_doc
		no_doc[num] = dc
		doc_no[dc] = num
		num += 1


def read_qry(path):
	global qrys_tf_list
	global no_qry
	os.chdir(path)
	file_names = os.listdir()
	dc = 0
	for file in os.listdir():
		if file.endswith(".txt"):
			file_path = f"{path}/{file}"
			with open(file_path, 'r') as f:
				doc_wd = []
				qryj_tf_dict = {}
				for line in f:
					for w in line.split():
						doc_wd.append(w)
						if w in qryj_tf_dict:
							qryj_tf_dict[w] += 1
						else:
							qryj_tf_dict[w] = 1
				dc += 1
			qrys_tf_list.append(qryj_tf_dict)				
	num = 0
	for dc in file_names: #create no_doc
		no_qry[num] = dc
		num += 1

def calculate_IDF(lexicon, TF_list): #words appear in how many docs
    # t11 = perf_counter()
    # idf_per_word = {}
    # for word in lexicon:
    #     for doc in TF_list:
    #         if word in doc:
    #             lexicon[word] += 1
    # t12 = perf_counter()
    # print("\ttime1: ", t12 - t11)
    # for word in lexicon:
    #     idf_per_word[word] = math.log((len(TF_list) - lexicon[word] + 0.5) / (lexicon[word] + 0.5))
    tmp2 = Path("/idf.txt").read_text()
    idf_per_word = json.loads(tmp2)
    return idf_per_word

def avg_len(docs_len_dict):
    sum1 = 0
    for doc in docs_len_dict:
        sum1 = sum1 + docs_len_dict[doc]
    return sum1/len(docs_len_dict)

def simDQ(docs_TF_list, qry_docs, lexiconIDF, docs_len, avg_doc_len, n):
    k1 = 8
    k3 = 1000
    b = 0.75
    delta = 0.357
    sim = {}
    num = 0
    #print(len(docs_tf_list))
    for doc in docs_TF_list:
        sumq = 0
        for word in qry_docs:

            if word in doc:
                tfprime = doc[word] / ((1 - b) + b * docs_len[no_doc[num]] / avg_doc_len)
                sumq += ((k1 + 1) * (tfprime + delta) / (k1 + tfprime + delta)) * ((k3 + 1) * qry_docs[word]) / (k3 + qry_docs[word]) * lexiconIDF[word] ###((k3 + 1) * qry_docs[word]) / (k3 + qry_docs[word])
        sim[no_doc[num]] = sumq
        num += 1
    result = dict(sorted(sim.items(), key = lambda item: item[1], reverse = True))
    return list(result.keys())[:n]

def topn_doc_eachQry(docs_tf_list, qrys_tf_list, idf_dict, docs_len_dict, avg_docs_len, n):
	new_qry_list = []
	for doc in qrys_tf_list:
		new_qry_list.append(simDQ(docs_tf_list, doc, idf_dict, docs_len_dict, avg_docs_len, n))
	return new_qry_list

def updateQry(qry_related_ndoc_list, docs_tf_list, n): 
	global  qrys_tf_list
	global nltk_stopwords
	a = 0.85
	qrys_tf_list1 = []
	dc = 0
	for lst in qry_related_ndoc_list: 
		tmp_doc_tf_dict = {}
		count_ori = 0
		for wname, wtimes in qrys_tf_list[dc].items():
			tmp_doc_tf_dict[wname] = wtimes * a
			count_ori += 1
		#print("add queries:", tmp_doc_tf_dict)
		for doc_name in lst:
			doc_num = doc_no.get(doc_name)
			#print("dndn: ", doc_name, doc_num)
			#doc_num = [k for k, v in no_doc.items() if v == doc_name]#int(no_doc.get(doc_name))
			#print("!!!",docs_tf_list[doc_num])
			for word_name, word_times in docs_tf_list[doc_num].items():
				if word_name in nltk_stopwords:
					continue
				if word_name in tmp_doc_tf_dict:
					tmp_doc_tf_dict[word_name] += word_times * (1 - a) / n
				else:
					tmp_doc_tf_dict[word_name] = word_times * (1 - a) / n
		#print("\nqr with all docs: ", json.dumps(tmp_doc_tf_dict)[:300])
		# count = 0
		# #print(json.dumps(tmp_doc_tf_dict)[:100])
		# for wname, wtimes in tmp_doc_tf_dict.items():
		# 	#print("wnwt: ",wname, wtimes, end = " ///")
		# 	if count >= count_ori:
		# 		tmp_doc_tf_dict[wname] = wtimes / n
		# 	count += 1
		#print(json.dumps(tmp_doc_tf_dict)[:100])
		qrys_tf_list1.append(tmp_doc_tf_dict)
		#print("qr with all docs1: ", json.dumps(tmp_doc_tf_dict)[:300])
		dc += 1
	
	return qrys_tf_list1

def final_sim(docs_tf_list, qrys_tf_list,  idf_dict, docs_len_dict, avg_docs_len):
	sim_qd = []
	for qry in qrys_tf_list:
		sim_qd.append(simDQ(docs_tf_list, qry, idf_dict, docs_len_dict, avg_docs_len, 1000))

	strlist = []
	n = 0
	for doc in sim_qd:
		s1 = no_qry[n][:-4] + ","
		for w in doc:
			s1 = s1 + w[:-4] + " "
		strlist.append(s1)
		n += 1	
	strlist = sorted(strlist)
	
	out_path = "./answer5-1.txt"
	f = open(out_path, "w")
	f.write("Query,RetrievedDocuments\n")
	for str1 in strlist:
		f.write(str1[:-1])
		f.write("\n")

# nlp = spacy.load('en_core_web_sm')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
# print('spaCy has {} stop words'.format(len(spacy_stopwords)))
top_n = 7
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
t0 = perf_counter()
doc_path = "/2021-ntust-information-retrieval-hw5/q_100_d_20000/data/docs/"
qry_path = "/2021-ntust-information-retrieval-hw5/q_100_d_20000/data/queries/"
read_doc(doc_path) #read docs
read_qry(qry_path)
t1 = perf_counter()
print("----files read---- time: ", t1 - t0)

num = 0
for i in sorted(word_set): #create word dict
	word_dict[num] = i
	word_dict_r[i] = num
	num += 1
t2 = perf_counter()
print("----word_dict built---- time: ", t2 - t1)

idf_dict = dict.fromkeys(sorted(word_set), 0)
idf_dict = calculate_IDF(idf_dict, docs_tf_list)
print(json.dumps(idf_dict)[:300])
t3 = perf_counter()
print("----idf_dict built---- time: ", t3 - t2)

avg_docs_len = avg_len(docs_len_dict)
qry_related_5doc_list = topn_doc_eachQry(docs_tf_list, qrys_tf_list, idf_dict, docs_len_dict, avg_docs_len, top_n)
print(json.dumps(qry_related_5doc_list)[:100])

qrys_tf_list = updateQry(qry_related_5doc_list, docs_tf_list, top_n)
print(json.dumps(qrys_tf_list)[:500])
t4 = perf_counter()
print("----query_tf updated---- time: ", t4 - t3)

final_sim(docs_tf_list, qrys_tf_list,  idf_dict, docs_len_dict, avg_docs_len)
t5 = perf_counter()
print("----get result---- time: ", t5 - t4)
print("time: ", t5 - t0)