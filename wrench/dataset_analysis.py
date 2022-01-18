from wrench.dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def calc_freqs(x):
  m = max(x)
  n = len(x)
  freqs = []
  for i in range(m+1):
    freqs.append(x.count(i)/n)
  return freqs

def determine_balance(dataname, dataset_path):
    """Make a histogram of the class balance of a dataset."""
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        dataname,
        extract_feature=False
    )
    f = []
    for data in [train_data, valid_data, test_data]:
        labels = data.labels
        freqs = calc_freqs(labels)
        f.append(freqs)
        # print(freqs)
    n_class = np.max(train_data.labels) + 1
    # print(n_class)

    # Plot chart
    width = 1/4
    ind = np.arange(n_class)
    labels=['train', 'val', 'test']
    for i, freqs in enumerate(f):
        plt.bar(ind + i*width, freqs, width, label=labels[i])
    plt.ylabel('Class proportion')
    plt.xticks(ind + width/2, [str(i) for i in range(n_class)])
    plt.xlabel('class')
    plt.title(f'Class proportion {dataname}')
    plt.legend()
    plt.show()
    plt.clf()

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
      # print(f'm:{majority}, {x.count(majority)}, {x.count(1-majority)}')
      # print(n_lfs)
      results.append(x.count(majority)/n_lfs)
  return results

def lf_maj_proportion(dataname, dataset_path):
    """Plot a histogram of the majority proportion per datapoint.
    The majority proportion is defined as the """
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        dataname,
        extract_feature=False
    )
    mps = []
    bins = np.linspace(0, 1, num=50)
    for data in [train_data, valid_data, test_data]:
        weak_labels = data.weak_labels
        mp = majority_proportion(weak_labels)
        mps.append(mp)
    labels=['train', 'val', 'test']
    colors=['red','yellow','blue']
    fig, axs = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        axs[i].hist(mps[i], label=labels[i], color=colors[i], bins=bins)
    axs[1].set_ylabel('Counts')
    axs[2].set_xlabel('Majority proportion')
    fig.suptitle(f'Majority proportion {dataname}')
    fig.legend()
    # fig.show()
    plt.show()
    fig.clear()

def effective_lf_count(dataname, dataset_path):
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        dataname,
        extract_feature=False
    )
    elf_counts = [[],[],[]]
    for i, data in enumerate([train_data, valid_data, test_data]):
        weak_labels = data.weak_labels
        elf_counts[i].append(len(weak_labels) - weak_labels.count(-1))
    labels=['train', 'val', 'test']
    colors=['red','yellow','blue']
    fig, axs = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        axs[i].hist(elf_counts[i], label=labels[i], color=colors[i])
    axs[1].set_ylabel('Counts')
    axs[2].set_xlabel('Effective number of labeling functions')
    fig.suptitle(f'Effective number of labeling functions {dataname}')
    fig.legend()
    # fig.show()
    plt.show()
    fig.clear()