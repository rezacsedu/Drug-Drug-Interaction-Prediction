from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, auc,average_precision_score
import numpy
import pandas as pd
import numpy as np

from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics

import random
import numbers
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import gc
import os

def getNegativeSize(all_combs, positive_pairs, n_propotion):
    x= len(all_combs) - len(positive_pairs) 
    if x < len(positive_pairs)*n_propotion:
         negative_size=x
    else:
        negative_size=len(positive_pairs)*n_propotion
        
    return negative_size

def getDataFrame(pairs, cls):
    pairs = np.array(pairs)
    if cls == 1:
        classes = np.ones(len(pairs))
    else:
        classes = np.zeros(len(pairs))

    data = list(zip(pairs[:,0],pairs[:,1],classes))
    df = pd.DataFrame(data,columns=['Drug1','Drug2','Class'])
    return df

def select_negative_samples(train_drugs, test_drugs, train_pairs, test_pairs_drugwise, test_pairs_pairwise, n_propotion):
    all_combs = set(itertools.combinations(sorted(train_drugs),2))
    # check whether there is enough pairs to be added as negatives
    
    negative_size=getNegativeSize(all_combs, train_pairs, n_propotion)

    unknowPairs = all_combs.difference(train_pairs)
    train_negatives =random.sample(unknowPairs, negative_size)
    
    all_combs = set([ tuple(sorted([drug1,drug2]))  for drug1 in train_drugs for drug2 in test_drugs])
    
    negative_size = getNegativeSize(all_combs, test_pairs_drugwise, n_propotion) 
    test_negatives_drugwise =random.sample(all_combs.difference(test_pairs_drugwise), negative_size)
    
    all_combs = set(itertools.combinations(sorted(test_drugs),2))  
    negative_size = getNegativeSize(all_combs, test_pairs_pairwise, n_propotion) 
    test_negatives_pairwise =random.sample(all_combs.difference(test_pairs_pairwise), negative_size)
    
    return train_negatives, test_negatives_drugwise, test_negatives_pairwise

def drugwise_k_fold_cross(all_drugs, all_pairs, n_fold ):
    n_subsets = int(len(all_drugs)/n_fold)
    subsets = dict()
    remain = set(all_drugs)
    for i in range(0,n_fold-1):
        subsets[i] = random.sample(remain, n_subsets)
        remain =remain.difference(subsets[i])
    subsets[n_fold-1]= list(remain)
    
    for i in reversed(range(0, n_fold)):
        test_drugs = subsets[i]
        train_drugs = []
        for j in range(0, n_fold):
            if i != j:
                train_drugs.extend(subsets[j])

        train_pairs = []
        for pair in all_pairs:
            drug1 = pair[0]
            drug2 = pair[1]
            if drug1 in train_drugs and drug2 in train_drugs:
                train_pairs.append(pair)
   
        test_pairs_drugwise =[]
        test_pairs_pairwise =[]
        for pair in all_pairs:
            drug1 = pair[0]
            drug2 = pair[1]
            if drug1 in test_drugs and drug2 in test_drugs:
                test_pairs_pairwise.append(pair)
            elif drug1 in test_drugs or drug2 in test_drugs:
                test_pairs_drugwise.append(pair)

        yield i,train_drugs, test_drugs, train_pairs, test_pairs_drugwise, test_pairs_pairwise 
        
        
def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores

def get_scores(clf, X_new, y_new):

    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
    #print(scorers)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores

def cv_run(run_index, fold_data, embedding_df, clfs):
    #print (fold_data)
    fold_index,train_drugs, test_drugs, train_positives, test_positives_drugwise, test_positives_pairwise = fold_data
    #print ("train drugs",len(train_drugs),"test drugs",len(test_drugs), file=f)
    train_negatives, test_negatives_drugwise, test_negatives_pairwise = select_negative_samples(train_drugs, test_drugs, train_positives, test_positives_drugwise, test_positives_pairwise, n_propotion=1)
    train =  pd.concat([getDataFrame(train_positives, 1),  getDataFrame(train_negatives, 0)],ignore_index=True) 

  
    test_drugwise =  pd.concat([getDataFrame(test_positives_drugwise, 1),  getDataFrame(test_negatives_drugwise, 0)],ignore_index=True)   
    test_pairwise =  pd.concat([getDataFrame(test_positives_pairwise, 1),  getDataFrame(test_negatives_pairwise, 0)],ignore_index=True) 
    train = train.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
   
    test_drugwise = test_drugwise.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
    test_pairwise = test_pairwise.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')



    features = train.columns.difference(['Drug1','Drug2' ,'Class', 'Drug_x', 'Drug_y'])
    X_train = train[features].values
    y_train = train['Class'].values

    X_test_drugwise =  test_drugwise[features].values
    y_test_drugwise = test_drugwise['Class'].values

    X_test_pairwise =  test_pairwise[features].values
    y_test_pairwise = test_pairwise['Class'].values

    drugwise_results = pd.DataFrame()
    pairwise_results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(X_train, y_train)
        scores2 = get_scores(clf, X_test_drugwise, y_test_drugwise)
        scores3 = get_scores(clf, X_test_pairwise, y_test_pairwise)
        scores2['method'] = name
        scores3['method'] = name
        scores2['fold'] = fold_index
        scores3['fold'] = fold_index
        scores2['run'] = run_index
        scores3['run'] = run_index
        drugwise_results = drugwise_results.append(scores2, ignore_index=True)
        pairwise_results = pairwise_results.append(scores3, ignore_index=True)
                                       
    return  drugwise_results,  pairwise_results                                   

def cross_validate_spark(sc, clfs, all_drugs, all_pairs, run_index, n_fold, embedding_df):    
    #print (cv)
    cv = drugwise_k_fold_cross(all_drugs, all_pairs, n_fold)
    rdd = sc.parallelize(cv).map(lambda fold_data: cv_run( run_index, fold_data, embedding_df, clfs ))
    all_scores = rdd.collect()
    return all_scores


def cross_validate(clfs, all_drugs, all_pairs, embedding_df, n_fold, f, n_propotion=1):

    drug_k_fold = drugwise_k_fold_cross(all_drugs, all_pairs, n_fold)
    #c1_results = pd.DataFrame()
    drugwise_results = pd.DataFrame()
    pairwise_results = pd.DataFrame()
    
    for i,(fold_data) in enumerate(drug_k_fold):
        #print (fold_data)
        train_drugs, test_drugs, train_positives, test_positives_drugwise, test_positives_pairwise = fold_data
        print ("train drugs",len(train_drugs),"test drugs",len(test_drugs), file=f)
        train_negatives, test_negatives_drugwise, test_negatives_pairwise = select_negative_samples(train_drugs, test_drugs, train_positives, test_positives_drugwise, test_positives_pairwise, n_propotion=1)
        train =  pd.concat([getDataFrame(train_positives, 1),  getDataFrame(train_negatives, 0)],ignore_index=True) 
       
    
        test_drugwise =  pd.concat([getDataFrame(test_positives_drugwise, 1),  getDataFrame(test_negatives_drugwise, 0)],ignore_index=True)   
        test_pairwise =  pd.concat([getDataFrame(test_positives_pairwise, 1),  getDataFrame(test_negatives_pairwise, 0)],ignore_index=True) 
        train = train.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
        
        test_drugwise = test_drugwise.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
        test_pairwise = test_pairwise.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
        features = train.columns.difference(['Drug1','Drug2' ,'Class', 'Drug_x', 'Drug_y'])
        X_train = train[features].values
        y_train = train['Class'].values

                             
        X_test_drugwise =  test_drugwise[features].values
        y_test_drugwise = test_drugwise['Class'].values

        X_test_pairwise =  test_pairwise[features].values
        y_test_pairwise = test_pairwise['Class'].values

        print("train positive set:", len(y_train[y_train==1]), " negative set:", len(y_train[y_train==0]), file=f)
        print("drugwise test positive set:",len(y_test_drugwise[y_test_drugwise==1])," negative set:",len(y_test_drugwise[y_test_drugwise==0]), file=f)
        print("pairwise test positive set:",len(y_test_pairwise[y_test_pairwise==1])," negative set:",len(y_test_pairwise[y_test_pairwise==0]), file=f)
        
        for name, clf in clfs:
            clf.fit(X_train, y_train)
            scores2 = get_scores(clf, X_test_drugwise, y_test_drugwise)
            scores3 = get_scores(clf, X_test_pairwise, y_test_pairwise)
            scores2['method'] = name
            scores3['method'] = name
            drugwise_results = drugwise_results.append(scores2, ignore_index=True)
            pairwise_results = pairwise_results.append(scores3, ignore_index=True) 
        
        display(drugwise_results)
        print("-----------------------------------------------------------------------------", file=f)

        del X_train, y_train, X_test_drugwise, y_test_drugwise, X_test_pairwise, y_test_pairwise
        del train_drugs, test_drugs, train_positives,test_positives_drugwise, test_positives_pairwise
        del train_negatives,test_negatives_drugwise, test_negatives_pairwise
        gc.collect()
        
    return drugwise_results, pairwise_results

def run_cv10(sc, clfs, embedding_df, commonDrugs, all_positives, n_fold, n_run, n_proportion, n_seed):

    drugwise_runs = pd.DataFrame()
    pairwise_runs = pd.DataFrame()


    bc_embedding_df = sc.broadcast(embedding_df)
    bc_positives = sc.broadcast(all_positives)
    bc_commonDrugs = sc.broadcast(commonDrugs)

    for i in range(n_run): 
        n_seed += i
        random.seed(n_seed)
        np.random.seed(n_seed)
        print ('run',i)
        all_scores =cross_validate_spark(sc, clfs, bc_commonDrugs.value, bc_positives.value, i, n_fold, bc_embedding_df.value)   
        drugwise = pd.DataFrame()
        pairwise = pd.DataFrame()

        for a in all_scores:
            drugwise= drugwise.append(a[0])
            pairwise= pairwise.append(a[1])

        drugwise_runs = drugwise_runs.append(drugwise)
        pairwise_runs = pairwise_runs.append(pairwise)
    
    return drugwise_runs, pairwise_runs

def getPositivePairs(drugbank_ddi, embedding_df):
    drugsInDrugbankDDI = set(drugbank_ddi['Drug1'].unique()).union(drugbank_ddi['Drug2'].unique())
    commonDrugs = drugsInDrugbankDDI.intersection(embedding_df.Drug.unique())
    print ('Drugs',len(commonDrugs))

    pairs = []
    classes = []

    ddiKnown = set([tuple(x) for x in  drugbank_ddi[['Drug1','Drug2']].values])

    for comb in itertools.combinations(sorted(commonDrugs),2):
        dr1=comb[0]
        dr2=comb[1]
        if (dr1,dr2)  in ddiKnown or  (dr2,dr1)  in ddiKnown:
            pairs.append((dr1,dr2))

    print("Postive size: %d" % len(pairs))
    all_positives = set(pairs)
    return commonDrugs, all_positives