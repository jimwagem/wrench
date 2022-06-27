from wrench.dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import csv

def majority_proportion(weak_labels):
    results = []
    for x in weak_labels:
        set_x = set(x)
        set_x.discard(-1)
        if set_x:
        # count non abstain label functions
            n_lfs=0
            for i in set_x:
                n_lfs += x.count(i)
            majority = max(set_x, key=x.count)
            results.append(x.count(majority)/n_lfs)
    n_single = results.count(1.0)
    return n_single/len(results)

def class_balance(labels, n_class):
    class_count = []
    for n in range(n_class):
        class_count.append(labels.count(n))
    n_points = sum(class_count)
    cb = [f"{i/n_points:.3f}" for i in class_count]
    return cb
    

if __name__=="__main__":
    path = '../../../datasets/'
    datasets = ['profteacher', 'imdb_12', 'imdb_136', 'amazon',
                'youtube', 'sms', 'census', 'yelp',
                'agnews', 'semeval', 'trec', 'chemprot']

    with open('./maj_prop.csv', 'w+') as f_maj_prop:
        with open('./class_balance.csv', 'w+') as f_cb:
            cb_writer = csv.writer(f_cb)
            maj_prop_writer = csv.writer(f_maj_prop)
            for ds in datasets:
                train_ds, valid_ds, test_ds = load_dataset(path, ds)
                wl = train_ds.weak_labels
                labels = train_ds.labels
                n_class = train_ds.n_class
                maj_prop = majority_proportion(wl)
                maj_prop_writer.writerow([ds, f"{maj_prop:.2f}"])
                cb = class_balance(labels, n_class)
                cb_writer.writerow([ds] + cb)