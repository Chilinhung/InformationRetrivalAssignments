# BM25 + SMM Rocchio
# KL-divergence similarity
############################

import os
import numpy as np
from numpy import zeros
import math
from math import log
from pylab import random
from time import perf_counter
import copy
import json
from pathlib import Path
import random
#word
word_set = set()
word_dict = {}
word_dict_r = {}
word_bg_dict = {}
idf_dict = {}

#doc
no_doc = {}
doc_no = {}
docs_tf_list = [] #{wordname: num}
unigram_list = []
docs_len_dict = {}
p_smm_w = []
docs_count = 20000
docs_totalword = 0
avg_docs_len = 0

#query
no_qry = {}
qrys_tf_list = []
qry_related_ndoc_list = []
qrys_len_dict = {}
qry_count = 100

# dic = {"123": 1}
# arr = random(3)
# print(arr[dic.get("123")] * 2)
# input()
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
						word_set.add(w)
				dc += 1
			qrys_tf_list.append(qryj_tf_dict)				
	num = 0
	for dc in file_names: #create no_doc
		no_qry[num] = dc
		num += 1

def getUnigram(docs_tf_list, docs_len_dict, no_doc):
	global unigram_list
	unigram_list = copy.deepcopy(docs_tf_list)
	dc = 0
	for doc in unigram_list:
		dname = no_doc.get(dc)
		dwc = docs_len_dict.get(dname)
		tmp = [dwc]*len(doc)
		#map(lambda x, y: x/y, doc.values(), tmp)
		#print(dwc, doc.values())
		for w in doc:
			doc[w] /= dwc
		dc += 1
def div_dk(x):
	global docs_totalword
	return x/docs_totalword

def getWord_BG(): #words appear how many times in all docs
	global word_bg_dict
	t11 = perf_counter()
	for word_name in word_bg_dict.keys():
		for doc in docs_tf_list:
			if word_name in doc:
				word_bg_dict[word_name] += doc.get(word_name)

def calculate_IDF(lexicon, TF_list): #words appear in how many docs

    t11 = perf_counter()
    idf_per_word = {}
    for word in lexicon:
        for doc in TF_list:
            if word in doc:
                lexicon[word] += 1
    t12 = perf_counter()
    print("\ttime1: ", t12 - t11)
    for word in lexicon:
        idf_per_word[word] = math.log((len(TF_list) - lexicon[word] + 2) / (lexicon[word] + 2))
    t13 = perf_counter()
    print("\ttime2: ", t13 - t12)
    tmp = json.dumps(idf_per_word)
    return idf_per_word

def avg_len(docs_len_dict):
    sum1 = 0
    for doc in docs_len_dict:
        sum1 = sum1 + docs_len_dict[doc]
    return sum1/len(docs_len_dict)

def simDQ(docs_TF_list, qry_docs, lexiconIDF, docs_len, avg_doc_len, n):
    k1 = 8
    k3 = 1 #0.9
    b = 0.75
    delta = 0.357
    sim = {}
    num = 0
    #print(len(docs_tf_list))
    for doc in docs_TF_list:
        sumq = 0
        for word in qry_docs:
            if(word in doc):
                tfprime = doc[word] / ((1 - b) * b * docs_len[no_doc[num]] / avg_doc_len)
                sumq += ((k1 + 1) * tfprime / (k1 + tfprime) + delta) * ((k3 + 1) * qry_docs[word]) / (k3 + qry_docs[word]) * lexiconIDF[word] ###
        sim[no_doc[num]] = sumq
        num += 1
    result = dict(sorted(sim.items(), key = lambda item: item[1], reverse = True))
    return list(result.keys())[:n]

def topn_doc_eachQry(docs_tf_list, qrys_tf_list, idf_dict, docs_len_dict, avg_docs_len, n):
	new_qry_list = []
	for doc in qrys_tf_list:
		new_qry_list.append(simDQ(docs_tf_list, doc, idf_dict, docs_len_dict, avg_docs_len, n))
	return new_qry_list
	
def SMM(word_set, word_bg_dict, word_dict, word_dict_r, docs_tf_list, totalword, n, qryi_related_ndoc_list, doc_no):
	alpha = 0.95
	cwidj = {}
	for dname in qryi_related_ndoc_list:
		dnum = doc_no[dname]
		for wname in docs_tf_list[dnum]:
			if wname not in cwidj:
				cwidj[wname] = docs_tf_list[dnum][wname]
			else:
				cwidj[wname] += docs_tf_list[dnum][wname]
	p_smm_w = dict.fromkeys(cwidj.keys(), random.random())
	p_tsmm_w = dict.fromkeys(cwidj.keys(), 0) #zeros(len(word_set))
	s = 0
	for wt in p_smm_w.values():
		s += wt
	for wk in p_smm_w.keys():
		p_smm_w[wk] /= s
	epoch = 100
	t00 = perf_counter()
	prevL = 0
	for e in range(epoch):
		print("Epoch:", e, end = "\r")
		#Estep
		for wk in p_tsmm_w.keys():
			p_tsmm_w[wk] = (1 - alpha) * p_smm_w[wk] / ((1 - alpha) * p_smm_w[wk] + alpha * (word_bg_dict[wk] / totalword))
		
		#Mstep
		s = 0
		for wk in cwidj.keys():
			s += cwidj[wk] * p_tsmm_w[wk]
		for wk in p_smm_w.keys():
			p_smm_w[wk] = cwidj[wk] * p_tsmm_w[wk] / s

		#L
		L = 0
		for doc in qryi_related_ndoc_list:
			dnum = doc_no[doc]
			for wk in p_smm_w.keys():
				if wk in docs_tf_list[dnum]:
					L += math.log((1 - alpha) * p_smm_w[wk] + alpha * word_bg_dict[wk] / totalword) * docs_tf_list[dnum][wk]
			#	L += (-1) * log(mult_by_wd)
		print("L: ", L)
		if abs(L - prevL) < 0.001:
			break
		prevL = L
	t02 = perf_counter()
	print("qry", n, " -----time: ", t02 - t00)
	return p_smm_w

def getSMM(word_set, word_bg_dict, word_dict, word_dict_r, docs_tf_list, totalword, qry_related_ndoc_list, doc_no):
	smm_list = []
	q = 0
	for qry in qrys_tf_list:
		print(q, len(qry_related_ndoc_list))
		smm_list.append(SMM(word_set, word_bg_dict, word_dict, word_dict_r, docs_tf_list, totalword, q, qry_related_ndoc_list[q], doc_no))
		q += 1
	return smm_list

def simUSK(docs_tf_list, qry, no_doc, p_smm_w, word_bg_dict, word_dict_r, totalword):
	a = 0.2
	b = 0.75
	r = 1
	sim_qi_dj = {}
	dc = 0
	#print("word8:",p_smm_w[8])
	for doc in docs_tf_list:
		sumqidjw = 0
		for word_name in qry.keys():
			if word_name in doc.keys():
				P_ULM_w = unigram_list[dc][word_name]
				P_SMM_w = p_smm_w.get(word_name, 0)
				P_BG_w = word_bg_dict[word_name] / totalword
				P_w_d = unigram_list[dc][word_name]
				#orgV = (-1) * ((a * unigram_list[dc][word_name] + (-1) * (math.log(b) + math.log(p_smm_w[w_no])) + (1 - a - b) * word_bg_dict[word_name] / totalword) * math.log(r + doc[word_name] + (1 - r) * word_bg_dict[word_name] / totalword))
				newV = (-1) * (a * P_ULM_w + b * P_SMM_w + (1 - a - b) * P_BG_w) * math.log(r * P_w_d + (1 - r) * P_BG_w)
				#print(orgV, newV)
				sumqidjw += newV
		sim_qi_dj[no_doc.get(dc)] = sumqidjw
		dc += 1
	rs = dict(sorted(sim_qi_dj.items(), key = lambda item: item[1], reverse = True))

	return list(rs.keys())[:1000]

def final_sim(unigram_list, p_smm_w, word_bg_dict, docs_tf_list, qrys_tf_list, word_dict, word_dict_r, no_qry, totalword):
	sim_qd = []
	q = 0
	for qry in qrys_tf_list:
		sim_qd.append(simUSK(docs_tf_list, qry, no_doc, p_smm_w[q], word_bg_dict, word_dict_r, totalword))
		q += 1

	strlist = []
	n = 0
	for doc in sim_qd:
		s1 = no_qry[n][:-4] + ","
		#print(no_qry[n])
		for w in doc:
			s1 = s1 + w[:-4] + " "
		strlist.append(s1)
		n += 1	
	strlist = sorted(strlist)
	
	out_path = "/Users/ansley/Desktop/answere1.txt"
	f = open(out_path, "w")
	f.write("Query,RetrievedDocuments\n")
	for str1 in strlist:
		f.write(str1[:-1])
		f.write("\n")


top_n = 5
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

getUnigram(docs_tf_list, docs_len_dict, no_doc)
print(json.dumps(docs_tf_list)[:45])
print(json.dumps(unigram_list)[:120])
t3 = perf_counter()
print("----unigram_list built---- time: ", t3 - t2)

word_bg_dict = dict.fromkeys(sorted(word_set), 0)
getWord_BG()
print(json.dumps(word_bg_dict)[:100])
t4 = perf_counter()
print("----word_background_dict(undividing totalword) built---- time: ", t4 - t3) #haven't div doc totalword

idf_dict = dict.fromkeys(sorted(word_set), 0)
idf_dict = calculate_IDF(idf_dict, docs_tf_list)
print(json.dumps(idf_dict)[:100])
t5 = perf_counter()
print("----idf_dict built---- time: ", t5 - t4)

avg_docs_len = avg_len(docs_len_dict)
qry_related_ndoc_list = topn_doc_eachQry(docs_tf_list, qrys_tf_list, idf_dict, docs_len_dict, avg_docs_len, top_n)

print("total words: ", docs_totalword)
p_smm_w = getSMM(word_set, word_bg_dict, word_dict, word_dict_r, docs_tf_list, docs_totalword, qry_related_ndoc_list, doc_no)
#print(json.dumps(p_smm_w)[:1000])
t7 = perf_counter()
print("----SMM built---- time: ", t7 - t5)

final_sim(unigram_list, p_smm_w, word_bg_dict, docs_tf_list, qrys_tf_list, word_dict, word_dict_r, no_qry, docs_totalword)
t8 = perf_counter()
print("----get result---- time: ", t8 - t7)
print("time: ", t8 - t0)