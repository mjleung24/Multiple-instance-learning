from cross_validation import cv_split
from build_data import generate_data
from clean_musk import clean_musk
from adaboost import AdaBoostMIL
from sklearn.metrics import accuracy_score

import numpy as np

def musk_results():
    X, Y = clean_musk()
    Y[Y!=1]=-1
    folds = cv_split(X, Y, folds=5)

    results = []
    
    for fold in folds:
        training_examples, training_labels, testing_examples, testing_labels = fold
        model = AdaBoostMIL(training_examples, training_labels, 'euclid', 0)
        model.fit(training_examples, training_labels)
        pred = model.boost_predict(testing_examples)
        accuracy = accuracy_score(testing_labels, pred)
        results.append(accuracy)
    
    return results

def artificial_results():
    np.random.seed(12345)
    X, Y = generate_data(bags_per_function=100, instances_per_bag=10)
    Y[Y!=1]=-1
    folds = cv_split(X, Y, folds=5)

    results = []

    for fold in folds:
        training_examples, training_labels, testing_examples, testing_labels = fold
        model = AdaBoostMIL(training_examples, training_labels, 'euclid', 0)
        model.fit(training_examples, training_labels)
        pred = model.boost_predict(testing_examples)
        accuracy = accuracy_score(testing_labels, pred)
        results.append(accuracy)

    return results

if __name__ == "__main__":
    function =  artificial_results()
    musk = musk_results()
    print("Results on function data:", function)
    print("Average Accuracy: ", np.mean(function), " Standard deviation of Accuracy: ", np.std(function))
    print("Results on musk data:", musk)
    print("Average Accuracy: ", np.mean(musk), " Standard deviation of Accuracy: ", np.std(musk))
