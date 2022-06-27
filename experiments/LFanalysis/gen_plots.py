from snorkel.labeling import LFAnalysis
from wrench.dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def polarity_hist(weak_labels, ds_name, save_file='./figs/'):
    wl = np.array(weak_labels)
    polarities = [len(np.unique(wl[:,i])) for i in range(wl.shape[1])]
    for i in range(wl.shape[1]):
        if -1 in np.unique(wl[:,i]):
            polarities[i] -= 1
    plt.hist(polarities)
    plt.xlabel('LF polarity')
    plt.ylabel('count')
    plt.title(f'LF polarities for {ds_name}')
    plt.savefig(save_file + f'{ds_name}_polarities.pdf')
    plt.clf()


def coverage_vs_accuracy_plot(weak_labels, true_labels, ds_name, save_file='./figs/', n_class=None):
    LFA = LFAnalysis(np.array(weak_labels))
    coverage = LFA.lf_coverages()
    accs = LFA.lf_empirical_accuracies(np.array(true_labels))
    plt.scatter(coverage, accs, alpha=0.5)
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.title(f'Coverage vs accuracy for {ds_name}')
    plt.grid()
    if n_class is not None:
        plt.axhline(y=1/n_class)
    plt.savefig(save_file + f'{ds_name}_covacc.pdf')
    # plt.show()
    plt.clf()


  

def low_coverage_accuracy_plot(weak_labels, true_labels, ds_name, save_file='./figs/'):
    LFA = LFAnalysis(np.array(weak_labels))
    coverage = LFA.lf_coverages()
    accs = LFA.lf_empirical_accuracies(np.array(true_labels))
    if len(accs[coverage < 0.01]) == 0:
        return
    plt.violinplot(accs[coverage < 0.01])
    plt.ylabel('Accuracy')
    plt.xticks(color='w')
    plt.xlabel('Coverage < 0.01')
    plt.title(f'Accuracy of LFs for {ds_name}')
    plt.savefig(save_file + f'{ds_name}_violin.pdf')
    # plt.show()
    plt.clf()

def effective_lf_count(weak_labels, ds_name, save_file='./figs/'):
    elf_counts = []
    for wl in weak_labels:
      elf_counts.append(len(wl) - wl.count(-1))
    bins = np.arange(np.max(elf_counts) + 1)
    plt.hist(elf_counts, bins=bins)
    plt.xticks(bins + 0.5, bins)
    plt.ylabel('Counts')
    plt.xlabel('Effective number of labeling functions')
    plt.title(f'Effective number of labeling functions {ds_name}')
    plt.savefig(save_file + f'{ds_name}_effeclf.pdf')
    # plt.show()
    plt.clf()

def LFA(dataset_name, path):
    train_ds, valid_ds, test_ds = load_dataset(path, dataset_name)
    labels = train_ds.labels
    weak_labels = train_ds.weak_labels
    n_class = train_ds.n_class
    coverage_vs_accuracy_plot(weak_labels, labels, dataset_name, n_class = n_class)
    effective_lf_count(weak_labels, dataset_name)
    low_coverage_accuracy_plot(weak_labels, labels, dataset_name)
    polarity_hist(weak_labels, dataset_name)


if __name__ == "__main__":
    path = '../../../datasets/'
    datasets = ['profteacher', 'imdb_12', 'imdb_136', 'amazon',
                'youtube', 'sms', 'census', 'yelp',
                'agnews', 'semeval', 'trec', 'chemprot']
    # datasets = ['census']
    for ds in datasets:
        LFA(ds, path)
