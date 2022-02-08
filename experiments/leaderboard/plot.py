import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

def histogram(datasets, models, metrics, labels=None):
  N = len(models)
  width = 1/(len(metrics) + 1)
  for dataset in datasets:
    ind = np.arange(N)
    for i, metric in enumerate(metrics):
      means = [np.mean(res_dict[(model,dataset,metric)]) for model in models]
      std = [np.std(res_dict[(model,dataset,metric)]) for model in models]
      plt.bar(ind + i*width, means, width, label=metric, yerr=std)
    plt.ylabel('metric value')
    plt.title(f'Results for {dataset}')
    if labels is None:
        labels = models
    plt.xticks(ind + width/len(metrics), labels)
    plt.legend(loc='best')
    plt.show()

if __name__=='__main__':
    path = '../../../datasets'
    bin_datasets = ['profteacher', 'imdb', 'imdb_136', 'amazon']
    multi_datasets = ['crowdsourcing']
    result_path = './results/results_final.csv'
    # binary_metrics = ['acc', 'auc', 'f1_binary', 'mcc']
    multi_metrics = ['acc', 'f1_macro', 'mcc']
    binary_metrics = ['acc','auc','f1_binary', 'mcc']
    models = ['Ground_truth_MLP','supervised_validation_MLP','2stage_MLP_snorkel','2stage_MLP_flyingsquid_triplet_median',
        '2stage_MLP_flyingsquid_triplet_mean', '2stage_MLP_MV', 'MLP_weasel']
    labels = ['G truth','validation','snorkel','fs_med', 'fs_mean',
        'MV', 'weasel']

    res_dict = defaultdict(lambda: [])
    with open(result_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            model = row[0]
            dataset = row[1]
            metric = row[2]
            val = float(row[3])
            res_dict[(model,dataset,metric)].append(val)
    
    histogram(bin_datasets, models, binary_metrics, labels)
    histogram(multi_datasets, models, multi_metrics, labels)