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

import ml
import itertools
import random
import numpy as np
import pandas as pd
import argparse

def runaKrepetition(all_negatives, positives_train, positives_test, embedding_df, nb_model, logistic_model, rf_model, n_run, n_seed):
    nb_scores_df = pd.DataFrame()
    lr_scores_df = pd.DataFrame()
    rf_scores_df = pd.DataFrame()

    for i in range(n_run): 
        n_seed += i
        random.seed(n_seed)
        np.random.seed(n_seed)
        print ('run',i)
        train, test = balance_data(positives_train, positives_test, all_negatives )
        train_df = train.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
        test_df = test.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')

        nb_scores, lr_scores, rf_scores = ml.crossvalid(train_df, test_df,  nb_model, logistic_model, rf_model)
        nb_scores_df = nb_scores_df.append(nb_scores, ignore_index=True)
        lr_scores_df = lr_scores_df.append(lr_scores, ignore_index=True)
        rf_scores_df = rf_scores_df.append(rf_scores, ignore_index=True)
    return nb_scores_df,lr_scores_df, rf_scores_df


def generateTrainTest(ddi_train, ddi_test, embedding_df):
    
    drugs_train = set(ddi_train.Drug1.unique())
    drugs_train = drugs_train.union(ddi_train.Drug2.unique())
    
    drugs_test = set(ddi_test.Drug1.unique())
    drugs_test = drugs_test.union(ddi_test.Drug2.unique())
    
    drugs = drugs_train.intersection(drugs_test)
    
    drugs = drugs.intersection(embedding_df.Drug.unique())
    #drugs = list(drugs)[:500]
    len(drugs)
    
    ddiKnownTest = set([tuple(x) for x in  ddi_test[['Drug1','Drug2']].values])
    ddiKnownTrain = set([tuple(x) for x in  ddi_train[['Drug1','Drug2']].values])

    pairs_train = list()
    classes_train = list()

    pairs_test = list()
    classes_test = list()


    pairs = list()
    classes = list()

    for dr1,dr2 in itertools.combinations(sorted(drugs),2):
        if dr1 == dr2: continue

        if (dr1,dr2)  in ddiKnownTrain or  (dr2,dr1)  in ddiKnownTrain:
            pairs_train.append((dr1,dr2))
            classes_train.append(1)
        elif (dr1,dr2)  in ddiKnownTest or  (dr2,dr1)  in ddiKnownTest: 
            pairs_test.append((dr1,dr2))
            classes_test.append(1) 
        else:
            pairs.append((dr1,dr2))
            classes.append(0)

    pairs_train = np.array(pairs_train)
    pairs_test = np.array(pairs_test)
    pairs = np.array(pairs)
    
    positives_train = pd.DataFrame(list(zip(pairs_train[:,0],pairs_train[:,1],classes_train)), columns=['Drug1','Drug2','Class'])
    positives_test = pd.DataFrame(list(zip(pairs_test[:,0],pairs_test[:,1],classes_test)), columns=['Drug1','Drug2','Class'])
    all_negatives = pd.DataFrame(list(zip(pairs[:,0],pairs[:,1],classes)), columns=['Drug1','Drug2','Class'])
    print ('Drugs',len(drugs))
    print("+|Train DDI size: ", len(positives_train))
    print("+|Test DDI size: ", len(positives_test)) 
    
    return positives_train, positives_test, all_negatives

def balance_data(positives_train, positives_test, all_negatives):
    
    negatives = all_negatives.sample(len(positives_train)+len(positives_test)) # for balanced class
    negative_train = negatives.iloc[:len(positives_train),:]
    negative_test = negatives.iloc[len(positives_train):,:]
    print("-|Train DDI size: ", len(negative_train))
    print("-|Test DDI size: ", len(negative_test))  

    train = pd.concat([positives_train, negative_train], ignore_index=True)
    test = pd.concat([positives_test, negative_test], ignore_index=True)
    
    return train, test


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', required=True, dest='embedding', help='enter path to embedding file')
    parser.add_argument('-test', required=True, dest='test', help='enter path to test')
    parser.add_argument('-train', required=True, dest='train', help='enter path to test')
    parser.add_argument('-o', required=True, dest='out', help='enter path to file for output')
    
    args = parser.parse_args()
    
    test_file = args.test
    train_file = args.train
    emb_file = args.embedding
    output = args.out
    
    ddi_test = pd.read_csv(test_file, sep='\t')
    ddi_train = pd.read_csv(train_file, sep='\t')
    
    embedding_df = pd.read_csv(emb_file, delimiter='\t')
    if 'Entity' in embedding_df.columns:
        embedding_df.Entity =embedding_df.Entity.str[-8:-1]
        embedding_df.rename(columns={'Entity':'Drug'}, inplace=True)
    
    positives_train, positives_test, all_negatives = generateTrainTest(ddi_train, ddi_test, embedding_df)
    
    nb_model = GaussianNB()
    lr_model = linear_model.LogisticRegression()
    rf_model = ensemble.RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1)
    
    n_seed =100
    n_run =10 
    n_proportion = 1


    k=10
    nb_scores_df,lr_scores_df, rf_scores_df = runaKrepetition(all_negatives, positives_train, positives_test, embedding_df, nb_model, lr_model, rf_model, n_run, n_seed)
    
    lr_scores_df['method']= 'Logistic Regression'
    nb_scores_df['method']= 'Naive Bayes'
    rf_scores_df['method']= 'Random Forest'
    all_scores_df = pd.DataFrame()
    all_scores_df = all_scores_df.append(lr_scores_df, ignore_index=True)
    all_scores_df = all_scores_df.append(nb_scores_df, ignore_index=True)
    all_scores_df = all_scores_df.append(rf_scores_df, ignore_index=True)
    print( all_scores_df.groupby('method').mean())
    
    all_scores_df.to_csv(output,sep=',', index=False)

