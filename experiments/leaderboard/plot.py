import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

def histogram(datasets, models, metrics, labels=None, save=False, baseline=None, baseline_std=None):
  plt.rcParams['figure.figsize']=(10,5)
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
    # Adjust figure size
    

    if labels is None:
        labels = models
    plt.xticks(ind + width/len(metrics), labels)
    # plt.tight_layout()
    plt.legend(loc='best')
    plt.grid()
    if baseline is not None:
        b = baseline[dataset]
        plt.axhline(y=b, c='red')
        if baseline_std is not None:
          b_std = baseline_std[dataset]
          plt.axhspan(b - b_std, b + b_std, alpha=0.2, color='red')
    if save:
        plt.savefig(f'./results/{dataset}.pdf')
    plt.show()

if __name__=='__main__':
    save=True
    path = '../../../datasets'

    # Weasel reproduction
    bin_datasets = ['profteacher', 'imdb_12', 'imdb_136', 'amazon']
    multi_datasets = ['crowdsourcing']
    result_path = './results/results_weasel.csv'
    multi_metrics = ['acc', 'f1_macro', 'mcc', 'ece']
    binary_metrics = ['auc','f1_max', 'mcc']
    models = ['Ground_truth_MLP','supervised_validation_MLP','2stage_MLP_snorkel',
        '2stage_MLP_flyingsquid_triplet_mean', '2stage_MLP_MV_hard','2stage_MLP_MV_soft', 'MLP_weasel_no_balance','MLP_weasel_balance']
    labels = ['G truth','validation','snorkel','fs_mean',
        'MV hard', 'MV soft', 'weasel', 'weasel cb']

    baseline = {
        'profteacher': 0.8698,
        'imdb_12' : 0.7722,
        'imdb_136' : 0.8210,
        'amazon' : 0.866
    }
    baseline_std = {
        'profteacher': 0.0046,
        'imdb_12' : 0.0045,
        'imdb_136' : 0.010,
        'amazon' : 0.0071
    }

    # Wrench datasets
    # bin_datasets = ['census','youtube','sms','yelp']
    # multi_datasets = ['agnews','trec','semeval','chemprot']
    # result_path = './results_wrench.csv'
    # multi_metrics = ['acc', 'mcc', 'ece']
    # binary_metrics = ['auc','f1_max', 'mcc', 'ece']
    # # binary_metrics = ['acc','auc','f1_binary','f1_max', 'mcc', 'ece']
    # models = ['Ground_truth_MLP','Optimal_vote_MLP','2stage_MLP_snorkel',
    #     '2stage_MLP_flyingsquid_triplet_mean','2stage_MLP_MV_soft', 'MLP_weasel_no_balance']
    # labels = ['G truth','Optimal vote','snorkel','fs_mean', 'MV soft', 'weasel']
    
    res_dict = defaultdict(lambda: [])
    with open(result_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            model = row[0]
            dataset = row[1]
            metric = row[2]
            val = float(row[3])
            res_dict[(model,dataset,metric)].append(val)
    
    histogram(bin_datasets, models, binary_metrics, labels, baseline=baseline, save=save, baseline_std=baseline_std)
    # histogram(multi_datasets, models, multi_metrics, labels)