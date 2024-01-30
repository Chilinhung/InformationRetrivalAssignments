# PLSA & VSM for doc similarity 
################################

import os
import collections
import numpy as np
from numpy import zeros, log
from pylab import random
import logging
import time
from time import perf_counter

#doc_wd_list = []
no_doc = {}
docs_tf_list = [] #times of doc words [{word name, times} {word name, times}, ...]
docs_wordcount = {}

no_qry = {}
qrys_tf_list = [] #times of query words
unigram_list = []
qrys_wordcount = {}

#qry_wd_list = []
query_names = []
qs_widjCount = []

word_set = set()
word_dict = {}

doc_count = 10000
topics = 64
alpha = 0.75


def read_doc(path):
	global docs_tf_list
	global word_set
	global docs_wordcount
	global no_doc
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
						doc_wd.append(w)
						if w in docj_tf_dict:
							docj_tf_dict[w] += 1
						else:
							docj_tf_dict[w] = 1
						word_set.add(w)
				docs_wordcount[file_names[dc]] = len(doc_wd) #doc j wordcount
				dc += 1
			docs_tf_list.append(docj_tf_dict)				
	num = 0
	for dc in file_names: #create no_doc
		no_doc[num] = dc
		num += 1

def read_qry(path):
	global qrys_tf_list
	global qrys_wordcount
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
				qrys_wordcount[file_names[dc]] = len(doc_wd) #doc j wordcount
				dc += 1
			qrys_tf_list.append(qryj_tf_dict)				
	num = 0
	for dc in file_names: #create no_doc
		no_qry[num] = dc
		num += 1

def getUnigram(docs_tf_list, docs_wordcount, no_doc):
	global unigram_list
	#print("1:\n" , docs_tf_list)
	unigram_list = docs_tf_list
	
	dc = 0
	for doc in unigram_list:
		dname = no_doc.get(dc)
		dwc = docs_wordcount.get(dname)
		for w in doc:
			doc[w] /= dwc
		dc += 1

def getResult(word_set, topics, doc_count, docs_tf_list, word_dict, no_doc, docs_wordcount):
	words_total = len(word_set)
	WiTk = random([topics, words_total])
	TkDj = random([doc_count, topics])
	#print("1:\n" , docs_tf_list)
	
	for t in range(topics):
		normalization = sum(WiTk[t, :])
		for i in range(words_total):
			WiTk[t, i] /= normalization
	for j in range(doc_count):
		normalization = sum(TkDj[j, :])
		for t in range(topics):
			TkDj[j, t] /= normalization

	#PLSA
	epoch = 20
	for e in range(epoch):
		print("Epoch: ", e+1)
		t0 = perf_counter()
		#Estep
		DjWiTk = []
		no_dc = 0
		count = 0
		for doc in docs_tf_list: #run words' tf in each doc
			dj = [] #temp 
			print(count, "/" , len(docs_tf_list), end = "\r")
			count += 1
			for i in doc: # each word in doc
				wt = [] #temp list of [word i, topic t, WiTk * TkDj]
				sum_t = 0 
				for t in range(topics): 	
					no_w = word_dict.get(i)
					tmp = WiTk[t][no_w] * TkDj[no_dc][t] # P(wi|Tk)P(Tk|dj)
					wt.append(tmp) #append P(wi|Tk)P(Tk|dj)
					sum_t += tmp  #ΣP(wi|Tk)P(Tk|dj)
				for t in range(topics):
					wt[t] /= sum_t  #p(tk|wi,dj)
				dj.append(wt)							
			no_dc += 1
			#print(dj)
			DjWiTk.append(dj)
		#print(DjWiTk)
		print("------Estep-------")
		t1 = perf_counter()
		print("time: ", t1 - t0)
	 	#Mstep
		WiTk = zeros([words_total * topics])
		for t in range(topics):
			d = 0
			c = 1
			for doc, doc1 in zip(docs_tf_list, DjWiTk):
				print("topic: ", t, "\t", c, "/" , len(DjWiTk), end = "\r")
				c += 1 
				w = 0
				sum_by_wddc = 0
				TkDj[d][t] = 0
				for wname, wtimes, wi in zip(doc.keys(),doc.values(), doc1):		
					no_w = word_dict.get(wname)	
					WiTk[t][no_w] = float(wtimes *  wi[t])
					sum_by_wddc += WiTk[t][no_w]
					TkDj[d][t] += WiTk[t][no_w]
				dname = no_doc.get(d)
				TkDj[d][t] /= docs_wordcount.get(dname)
				for wname in doc.keys():
					no_w = word_dict.get(wname)
					WiTk[t][no_w] /= sum_by_wddc
				#print(WiTk)
			#print()
		#print(WiTk)
		#print(TkDj)
		print("------Mstep-------")
		t2 = perf_counter()
		print("time: ", t2 - t1)
		L = 0
		no_dc = 0
		for doc in docs_tf_list: #run words' tf in each doc
			for i in doc: # each word in doc
				sum_t = 0 
				for t in range(topics): 
					no_w = word_dict.get(i)
					sum_t += WiTk[t][no_w] * TkDj[no_dc][t] # P(wi|Tk)P(Tk|dj)
				#print(doc.get(i)," , ", sum_t, end = "\t|\t")
				L += doc.get(i) + log(sum_t)
			#print(L)
			no_dc += 1					
		print("-------L: ", abs(L), end = "--------\n")
		t3 = perf_counter()
		print("time: ", t3 - t2)

	getUnigram(docs_tf_list, docs_wordcount, no_doc)
	#𝑃(𝑞/𝑑𝑗)
	p_qd_list = []
	c = 1
	for qry in qrys_tf_list: # each query 
		print(c,"/",len(qrys_tf_list), end = "\r")
		c+=1
		p_qdj_dict = {}
		no_dc = 0
		for doc in unigram_list: # each doc
			p_qdj = 0
			pwidj = 0
			for qw in qry: # each query word
				if qw in doc.keys(): # if query wi in dj
					sum_by_tpk = 0
					for t in range(topics):
						no_w = word_dict.get(qw)
						sum_by_tpk += WiTk[t][no_w] * TkDj[no_dc][t]
					pwidj = log(alpha) + log(doc.get(qw)) + log(1-alpha) + log(sum_by_tpk)
					#print(pwidj, alpha, doc.get(qw), sum_by_tpk)
				if pwidj != 0:
					p_qdj += log(abs(pwidj)) ####
			p_qdj_dict[no_doc.get(no_dc)] = p_qdj
			no_dc += 1
		#print(p_qdj_dict)
		#rank related docs
		result = dict(sorted(p_qdj_dict.items(), key = lambda item: item[1], reverse = True)) 
		#print("result:\n", result)
		#get top 3000 according to unsorted queries
		p_qd_list.append(list(result.keys())[:3000])
	#print(p_qd_list)

	out_path = "/Users/ansley/Desktop/answer.txt"
	f = open(out_path, "w")
	str_list = []
	q = 0
	for qry in p_qd_list:
		s1 = no_qry[q][:-4] + ","
		for doc in qry:
			s1 = s1 + doc[:-4] + " "
		str_list.append(s1)
		q += 1
	str_list = sorted(str_list)
	print("string list: ")
	#print(str_list)

	f.write("Query,RetrievedDocuments\n")
	s = 0
	for str1 in str_list:
		f.write(str1[:-1])
		f.write("\n")

doc_path = "./2021-ntust-information-retrieval-hw4/q_100_d_10000/data/docs/"
qry_path = "./2021-ntust-information-retrieval-hw4/q_100_d_10000/data/queries/"
read_doc(doc_path) #read docs
read_qry(qry_path)
#print(docs_tf_list)
#print(qrys_tf_list)
#print(docs_wordcount)
#print(qrys_wordcount)
print("----files read-----")
num = 0
for i in sorted(word_set):
	word_dict[i] = num
	num += 1
#print(word_dict)

print("----word_dict----")
getResult(word_set, topics, doc_count, docs_tf_list, word_dict, no_doc, docs_wordcount)
