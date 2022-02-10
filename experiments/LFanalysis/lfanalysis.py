from snorkel.labeling import LFAnalysis
from wrench.dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np


def coverage_vs_accuracy_plot(weak_labels, true_labels, ds_name, save_file='./figs/'):
    LFA = LFAnalysis(np.array(weak_labels))
    coverage = LFA.lf_coverages()
    accs = LFA.lf_empirical_accuracies(np.array(true_labels))
    plt.scatter(coverage, accs)
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.title(f'Coverage vs accuracy for {ds_name}')
    plt.savefig(save_file + f'{ds_name}.pdf')
    # plt.show()
    plt.clf()


def low_coverage_accuracy_plot(weak_labels, true_labels, ds_name, save_file='./figs/'):
    LFA = LFAnalysis(np.array(weak_labels))
    coverage = LFA.lf_coverages()
    accs = LFA.lf_empirical_accuracies(np.array(true_labels))
    plt.violinplot(accs[coverage < 0.01])
    plt.ylabel('Accuracy')
    plt.xticks(color='w')
    plt.xlabel('Coverage < 0.01')
    plt.title(f'Accuracy of LFs for {ds_name}')
    plt.savefig(save_file + f'{ds_name}_violin.pdf')
    # plt.show()
    plt.clf()


def LFA(dataset_name, path):
    train_ds, valid_ds, test_ds = load_dataset(path, dataset_name)
    labels = train_ds.labels
    weak_labels = train_ds.weak_labels
    coverage_vs_accuracy_plot(weak_labels, labels, dataset_name)
    low_coverage_accuracy_plot(weak_labels, labels, dataset_name)


if __name__ == "__main__":
    path = '../../../datasets/'
    datasets = ['profteacher', 'imdb', 'imdb_136', 'amazon', 'crowdsourcing']
    for ds in datasets:
        LFA(ds, path)
