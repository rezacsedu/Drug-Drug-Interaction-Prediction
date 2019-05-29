import argparse
import gzip, os, csv
import numpy as np
import random
import time
import networkx as nx

import findspark
findspark.init("/home/rkarim/spark-2.3.1-bin-hadoop2.7/")

from pyspark import SparkConf, SparkContext

if False: 
    sc.stop()

config = SparkConf()
config.setMaster("local[5]")
config.set("spark.executor.memory", "8g")
config.set('spark.driver.memory', '12g')
config.set("spark.memory.offHeap.enabled",True)
config.set("spark.memory.offHeap.size","20g") 
sc = SparkContext(conf=config)
print (sc)

def addTriple(net, source, target, edge):
    if source in net:
        if  target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target]= set([edge])
    else:
        net[source]={}
        net[source][target] = set([edge])
    
    return net
            
def getLinks(net, source):
    if source not in net:
        return {}
    return net[source]

def randomWalkUniform(triples, startNode, max_depth=5):
    next_node =startNode
    path = 'n'+str(startNode)+'->'
    for i in range(max_depth):
        neighs = getLinks(triples,next_node)
        #print (neighs)
        if len(neighs) == 0: break
        weights = []
        queue = []
        for neigh in neighs:
            for edge in neighs[neigh]:
                queue.append((edge,neigh))
                weights.append(1.0)
        edge, next_node = random.choice(queue)
        path = path+ 'e'+str(edge)+'->'
        path = path+ 'n'+str(next_node)+'->'

    return path

def preprocess(folders, filename):
    entity2id = {}
    relation2id = {}
    triples = {}

    ent_counter = 0
    rel_counter = 0
    for dirname in folders:
        for fname in os.listdir(dirname):
            if not filename in fname: continue
            print (fname)
            gzfile= gzip.open(os.path.join(dirname, fname), mode='rt', encoding='utf-8')

            for line in csv.reader(gzfile, delimiter=' ', quotechar='"'):
                h = line[0]
                r = line[1]
                t = line[2]          

                if not t.startswith('<'): continue
                if 'ddi-interactor-in' in r: 
                     continue

                if h in entity2id:
                    hid = entity2id[h]
                else:
                    entity2id[h] = ent_counter
                    ent_counter+=1
                    hid = entity2id[h]

                if t in entity2id:
                    tid = entity2id[t]
                else:
                    entity2id[t] = ent_counter
                    ent_counter+=1
                    tid = entity2id[t]

                if r in relation2id:
                    rid = relation2id[r]
                else:
                    relation2id[r] = rel_counter
                    rel_counter+=1
                    rid = relation2id[r]
                addTriple(triples, hid, tid, rid)
            print ('Relation:',rel_counter, ' Entity:',ent_counter)
    return entity2id,relation2id,triples

folders = ['/home/rkarim/DDI/data/drugbank/v5/']
fileext = 'nq.gz'
entity2id, relation2id, triples = preprocess(folders, fileext)

num_triples=0
for source in triples:
    for  target in triples[source]:
        num_triples+=len(triples[source][target])
print ('Number of triples',num_triples)

walks = 5
path_depth = 10
paths = randomNWalkUniform(triples, 100, walks, path_depth)
print('\n'.join(paths))

entities = list(entity2id.values())
b_triples = sc.broadcast(triples)

import os
os.environ['HADOOP_HOME'] = "C:/hadoop"

folder = 'C:/Users/admin-karim/Downloads/GraphEmbedding4DDI-master/GraphEmbedding4DDI-master/data/walks5/'
#if not os.path.isdir(folder):
    #os.mkdir(folder)
walks = 250
maxDepth = 5
for path_depth in range(1,maxDepth):
    filename = folder+'randwalks_n%d_depth%d_pagerank_uniform.txt'%(walks, path_depth)
    print (filename)
    start_time =time.time()
    rdd = sc.parallelize(entities).flatMap(lambda n: randomNWalkUniform(b_triples.value, n, walks, path_depth)).persist()
    rdd.saveAsTextFile(filename)
    elapsed_time = time.time() - start_time
    print ('Time elapsed to generate features:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
def saveData(entity2id, relation2id, triples, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)  
    
    entity2id_file= open(os.path.join(dirname, 'entity2id.txt'),'w',  encoding='utf-8')
    relation2id_file = open(os.path.join(dirname, 'relation2id.txt'),'w',  encoding='utf-8')
    train_file = open(os.path.join(dirname, 'train2id.txt'),'w',  encoding='utf-8')

    train_file.write(str(num_triples)+'\n') 
    for source in triples:
        for  target in triples[source]:  
            hid=source
            tid =target
            for rid  in triples[source][target]:
                train_file.write("%d %d %d\n"%(hid,tid,rid))

    entity2id_file.write(str(len(entity2id))+'\n')  
    for e in sorted(entity2id, key=entity2id.__getitem__):
        entity2id_file.write(e+'\t'+str(entity2id[e])+'\n')  

    relation2id_file.write(str(len(relation2id))+'\n')    
    for r in sorted(relation2id, key=relation2id.__getitem__):
        relation2id_file.write(r+'\t'+str(relation2id[r])+'\n') 
        
    train_file.close()
    entity2id_file.close()
    relation2id_file.close()

dirname = 'DB5/'
saveData(entity2id, relation2id, triples, dirname)    
    
import gensim
class MySentences(object):
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename

    def __iter__(self):
        print ('Processing ',self.filename)
        for subfname in os.listdir(self.dirname):
            if not self.filename in subfname: continue
            fpath = os.path.join(self.dirname, subfname)
            for fname in os.listdir(fpath):
                if not 'part' in fname: continue
                if '.crc' in fname: continue
                try:
                    for line in open(os.path.join(fpath, fname), mode='r'):
                        line = line.rstrip('\n')
                        words = line.split("->")
                        yield words
                except Exception:
                    print("Failed reading file:")
                    print(fname)

def extractFeatureVector(model, drugs, id2entity, output): 
  
    header="Entity"
    ns = "n"
    first = ns+str(drugs[0])

    for i in range(len(model.wv[first])):
        header=header+"\tfeature"+str(i)
        
    fw=open(output,'w')
    fw.write(header+"\n")

    for id_ in sorted(drugs):
        nid =ns+str(id_)
        if  (nid) not in  model.wv:
            print (nid)
            continue
        vec = model.wv[nid]
        vec = "\t".join(map(str,vec))
        fw.write(id2entity[id_]+'\t'+str(vec)+'\n')
    fw.close()

maxDepth = 5

def trainModel(drugs, id2entity, datafilename, model_output, vector_output, pattern, maxDepth):
    
    if not os.path.isdir(model_output):
        os.mkdir(model_output)
        
    if not os.path.isdir(vector_output):
        os.mkdir(vector_output)
    
    output = model_output + pattern +'/'
    if not os.path.isdir(output):
        os.mkdir(output)
    
    sentences = MySentences(datafilename, filename=pattern) # a memory-friendly iterator
    word2vecModel = gensim.models.Word2Vec(size=300, workers=10, window=5, sg=1, negative=15, iter=20)

    word2vecModel.build_vocab(sentences)
    corpus_count = word2vecModel.corpus_count
    del word2vecModel
    
    #sg/cbow features iterations window negative hops random walks
    sgModel = gensim.models.Word2Vec(size=300, workers=10, window=5, sg=1, negative=15, iter = 20)
    sgModel.build_vocab(sentences)

    sgModel.train(sentences, total_examples = corpus_count, epochs = 10)
    modelName = 'RDF2Vec_full_sg_300_5_5_15_2_500' + '_d' + str(maxDepth)
    sgModel.save(output + modelName)
    
    extractFeatureVector(sgModel, drugs, id2entity, vector_output + modelName +'_' + pattern + '.txt')
    del sgModel
    
    #cbow 300
    cbowModel = gensim.models.Word2Vec(size=300, workers=10, window=5, sg=0, iter=20,cbow_mean=1, alpha = 0.05)
    cbowModel.build_vocab(sentences)

    cbowModel.train(sentences, total_examples=corpus_count, epochs = 10)
    modelName = 'RDF2Vec_full_cbow_300_5_5_2_500'+'_d'+str(maxDepth)
    cbowModel.save(output+ modelName)
    extractFeatureVector(cbowModel, drugs, id2entity, vector_output+modelName +'_'+pattern+'.txt')
    del cbowModel   

import pandas as pd
ddi_df = pd.read_csv('/home/rkarim/DDI/input/result.csv',sep=',')

db_ns ='http://bio2rdf.org/drugbank:'
ddi_df.Drug1 = '<'+db_ns+ddi_df.Drug1+'>'
ddi_df.Drug2 = '<'+db_ns+ddi_df.Drug2+'>'

db_entities = set()
drugs = set(ddi_df.Drug1.unique()).union(ddi_df.Drug2.unique())
for dbid in drugs:
    if dbid in entity2id:
        db_entities.add(entity2id[dbid])

db_entities =list(db_entities)
print(len(db_entities))
id2entity = {value:key for key,value in entity2id.items()} 

datafilename = '/home/rkarim/DDI/data/walks5/'
model_output = '/home/rkarim/DDI/models/RDF2Vec_model/'    
pattern = 'uniform'
vector_output =  '/home/rkarim/DDI/vectors/RDF2Vec/'
trainModel(db_entities, id2entity, datafilename, model_output, vector_output, pattern, maxDepth)

