import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":
    files = ['./results/multi_gauss_snorkel.csv', './results/multi_gauss_flyingsquid.csv', './results/multi_gauss_weasel.csv']
    names=['snorkel', 'flyingsquid', 'weasel']
    bad_results = []
    random_results = []
    constant_results = []
    n_good_lf = 5
    for f in files:
        bad_file_results = []
        random_file_results = []
        constant_file_results = []
        idxs = set()
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                t=row[0]
                idx = int(row[1])
                mean=float(row[2])
                std=float(row[3])
                if t=='bad':
                    bad_file_results.append((mean, std))
                elif t=='random':
                    random_file_results.append((mean, std))
                elif t=='constant':
                    constant_file_results.append((mean, std))
                idxs.add(idx)
            bad_results.append(bad_file_results)
            random_results.append(random_file_results)
            constant_results.append(constant_file_results)

    x_range = list(sorted(idxs))
    fig, axs = plt.subplots(3, 1, sharex=True)
    for name, results in zip(names, bad_results):
        means = np.array([e[0] for e in results])
        stds = np.array([e[1] for e in results])
        axs[0].plot(x_range, means, label=name)
        axs[0].fill_between(x_range, means+stds, means-stds, alpha=0.3)
        axs[0].set_ylim([0, 1.05])

    for name, results in zip(names, random_results):
        means = np.array([e[0] for e in results])
        stds = np.array([e[1] for e in results])
        axs[1].plot(x_range, means, label=name)
        axs[1].fill_between(x_range, means+stds, means-stds, label=name, alpha=0.3)
        axs[1].set_ylim([0, 1.05])
    for name, results in zip(names, constant_results):
        means = np.array([e[0] for e in results])
        stds = np.array([e[1] for e in results])
        axs[2].plot(x_range, means, label=name)
        axs[2].fill_between(x_range, means+stds, means-stds, label=name, alpha=0.3)
        axs[2].set_ylim([0, 1.05])
        axs[2].set_xticks(x_range)

    # Plot ratio of good lfs on top
    top = axs[0].twiny()
    top.set_xticks(axs[2].get_xticks())
    top.set_xticklabels([np.round(n_good_lf / (n_good_lf + v), 2) for v in axs[2].get_xticks()])

    axs[0].legend()
    axs[1].set_ylabel('Accuracy')
    axs[2].set_xlabel('Number of adversarial labeling functions')

    axs[0].set_title('Orthogonal task')
    axs[1].set_title('Random')
    axs[2].set_title('Constant')
    fig.suptitle('Effect of adversarial labeling function on classification')
    plt.tight_layout()
    plt.savefig("./results/multi_gaussian.png")
    plt.show()
