import csv
import numpy as np
import sys
import pandas as pd
import itertools
import math
import time
from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import networkx as nx
import random
import numbers
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from src import ml

import findspark
findspark.init("/home/spark-2.3.1-bin-hadoop2.7/")

from pyspark import SparkConf, SparkContext

if False: 
    sc.stop()

config = SparkConf()
config.setMaster("local[8]")
config.set("spark.executor.memory", "56g")
config.set('spark.driver.memory', '30g')
config.set("spark.memory.offHeap.enabled",True)
config.set("spark.memory.offHeap.size","64g") 
sc = SparkContext(conf=config)

ddi_df = pd.read_csv("full_DDI.txt", sep='\t')
ddi_df.head()

featureFilename = "vectors/RDF2Vec/RDF2Vec_sg_300_5_5_15_2_500_d5_uniform.txt" 
embedding_df = pd.read_csv(featureFilename, delimiter='\t') 

embedding_df.Entity = embedding_df.Entity.str[-8:-1]
embedding_df.rename(columns={'Entity':'Drug'}, inplace=True)

len(set(ddi_df.Drug1.unique()).union(ddi_df.Drug2.unique()) )
pairs, classes = ml.generatePairs(ddi_df, embedding_df)

reg = L1L2(l1=0.01, l2=0.01)
model = ml.Conv_LSTM(num_classes=2, timesteps=8, reg=reg)
model.summary()
convLSTM, history = ml.model_train(model, number_epoch=20)

kNN = KNeighborsClassifier(n_neighbors=3)
NB = GaussianNB()
SVM = svm.SVC()
LR = linear_model.LogisticRegression(C=0.01)
RF = ensemble.RandomForestClassifier(n_estimators=5, n_jobs=-1)
GBT = ensemble.GradientBoostingClassifier(n_estimators=5, max_leaf_nodes=3,max_depth=3)

clfs = [('LR',LR),('SVM',SVM),('NB',NB),('KNN',KNN),('GBT',GBT),('RF',LR),('Conv-LSTM Regression',convLSTM)]

n_seed =100
n_fold =5 
n_run =10
n_proportion = 1,
all_scores_df = ml.kfoldCV(sc, pairs, classes, embedding_df, clfs, n_run, n_fold, n_proportion, n_seed)

all_scores_df.groupby(['method','run']).mean().groupby('method').mean()
all_scores_df.to_csv('Results.txt',sep='\t', index=False)
