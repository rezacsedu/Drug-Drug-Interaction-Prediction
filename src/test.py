from disjoint import *

def test_balance_data_and_get_cv_twofold_doublenegative_disjoint():
    #drugs = ["c1", "c2", "c3", "c4",  "c5", "c6"]
    drugs = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    #drug_to_values = dict([("c1", [[1,0,0]]), ("c2", [[1,1,0]]), ("c3", [[1,0,1]]), ("c4", [[0,0,1]])])
    pairs_sample = [ ['c1','c5'],  ['c1','c6'],  ['c1','c4'],  ['c2','c5'],  ['c2','c6'],  ['c2','c3'],  ['c3','c8'],  ['c3','c6'], \
                    ['c4','c5'], ['c4','c7'], ['c6','c7'] ]
    #pairs_sample = list(set([ tuple(sorted((dr1, dr2))) for dr2 in drugs for dr1 in drugs  if dr1 != dr2]))
    
    pairs_sample = set( tuple(pair) for pair in pairs_sample)
    print (pairs_sample)
    classes_sample = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0] # 2 known drug-disease associations
    #classes_sample = [1, 1, 1, 0, 0, 1]
    print (len(classes_sample))
    n_fold = 2
    n_proportion = 1
    drug_k_fold = drugwise_k_fold_cross(drugs, pairs_sample, n_fold)
    
    for i,(fold_data) in enumerate(drug_k_fold):
        train_drugs, test_drugs, train_positives, test_positives_C2, test_positives_C3 = fold_data
        print ("train drugs",train_drugs,"test drugs",test_drugs)
        train_negatives, test_negatives_C2, test_negatives_C3 = select_negative_samples(train_drugs, test_drugs, train_positives, test_positives_C2, test_positives_C3, n_propotion=1)
        print("train_positives", train_positives ,"train_negatives", train_negatives)
        print("test_positives_C2", test_positives_C2,"test_negatives_C2", test_negatives_C2)
        print("test_positives_C3", test_positives_C3, "test_negatives_C3", test_negatives_C3)
        print ('======================')

n_seed =10
random.seed(n_seed)
np.random.seed(n_seed)
test_balance_data_and_get_cv_twofold_doublenegative_disjoint()