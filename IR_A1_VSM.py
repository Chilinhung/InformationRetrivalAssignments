# Vector Space Model for doc similarity
###################################

import os
import math
import json

lexSet = set()
docs_TF_list = []
qry_TF_list = []

#read each file & calculate tf of docs & create lexiconSet 
def rf_caldoc_crlexst(file_path): 
    with open(file_path, 'r') as f:
        doci_tf_dic = {}
        for line in f:
            for word in line.split():
                if(word in doci_tf_dic):
                    doci_tf_dic[word] += 1
                else:
                    doci_tf_dic[word] = 1
                lexSet.add(word) 
    return doci_tf_dic
#read docs/ read query

def read_doc_files():
    path = './2021-ntust-information-retrieval-hw1/data/docs/'
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            doc_dict = rf_caldoc_crlexst(file_path) #read doc
            docs_TF_list.append(doc_dict)
    doc_names = os.listdir()
    return doc_names

def read_qry_files():
    path1 = './2021-ntust-information-retrieval-hw1/data/queries/'
    os.chdir(path1)
    for file1 in os.listdir():
        if file1.endswith(".txt"):
            file_path1 = f"{path1}/{file1}"
            qry_TF_list.append(rf_caldoc_crlexst(file_path1))
    qry_names = os.listdir()
    return qry_names
    
def calculate_IDF(lexicon, TF_list):
    idf_per_word = {}
    for word in lexicon:
        for doc in TF_list:
            if(word in doc):
                lexicon[word] += 1
    for word in lexicon:
        idf_per_word[word] = math.log(len(TF_list) / lexicon[word])
        
    sortidf = dict(sorted(idf_per_word.items(), key =lambda item: item[1], reverse = True))
    print(sortidf)
    return idf_per_word

def getQueryTF(qry_TF_list):
    for doc in qry_TF_list:
        mx = 0
        for word in doc:
            if (doc[word] > mx):
                mx = doc[word]
        for word in doc:
            doc[word] = 0.5 + 0.5 * (doc[word]/mx)
    return qry_TF_list

def getWeight(tf, idf):
    weight = []
    for doc in tf:
        tmpdict = {}
        for word in doc:
            if(word in idf):
                tmpdict[word] = doc[word]*math.pow(idf[word], 2.4)             
            else:
                tmpdict[word] = 0
        weight.append(tmpdict)
    return weight 

def simDQ(wd, wq_doc):
    sim = {}
    no = 0
    for doc1 in wd:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for word in doc1:
            sum3 += doc1[word] * doc1[word]
        for word in wq_doc: # each word in wq's each doc
            if(word in doc1):
                sum1 += wq_doc[word] * doc1[word]
            sum2 += wq_doc[word] * wq_doc[word]
        s4 = math.sqrt(sum2) * math.sqrt(sum3)
        if(s4 != 0):
            sim[doc_names[no]] = sum1 / s4
        else:
            sim[doc_names[no]] = 0
        no += 1
    result = dict(sorted(sim.items(), key = lambda item: item[1], reverse = True))
    print(json.dumps(result)[:500])
    return list(result.keys())[:1000]

def getSim_outputResult(wd, wq):
    # make result dictlist
    rs = []
    for doc in wq:
        rs.append(simDQ(wd, doc))

    #make output stringlist
    path2 = "./test.txt"
    strlist = []
    r = 0
    f = open(path2,"w")
    for doc in rs:
        #print(r)
        str1 = qry_names[r][:-4] + ","
        r += 1
        for w in doc:
            str1 = str1 + w[:-4] + " "
        strlist.append(str1)
    strlist = sorted(strlist)
    print(type(strlist))
    print(strlist)

    #write to file
    f.write("Query,RetrievedDocuments\n")
    for str1 in strlist:
        f.write(str1[:-1])
        f.write("\n")

doc_names = read_doc_files()
#print(len(doc_names))
#print(len(docs_TF_list))
#print(len(qry_names))
#print(len(qry_TF_list))
lexicon = dict.fromkeys(lexSet, 0)
lexiconIDF = calculate_IDF(lexicon, docs_TF_list)
qry_names = read_qry_files()
wd = getWeight(docs_TF_list, lexiconIDF)
qry_TF_list = getQueryTF(qry_TF_list)
wq = getWeight(qry_TF_list, lexiconIDF)
getSim_outputResult(wd, wq)





