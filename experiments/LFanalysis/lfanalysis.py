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


def LFA(dataset_name, path):
    train_ds, valid_ds, test_ds = load_dataset(path, dataset_name)
    labels = train_ds.labels
    weak_labels = train_ds.weak_labels
    coverage_vs_accuracy_plot(weak_labels, labels, dataset_name)
    low_coverage_accuracy_plot(weak_labels, labels, dataset_name)
    polarity_hist(weak_labels, dataset_name)


if __name__ == "__main__":
    path = '../../../datasets/'
    # datasets = ['profteacher', 'imdb', 'imdb_136', 'amazon', 'crowdsourcing']
    datasets = ['census']
    for ds in datasets:
        LFA(ds, path)
