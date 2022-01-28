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
    elf_counts = []
    weak_labels = train_data.weak_labels
    for wl in weak_labels:
      elf_counts.append(len(wl) - wl.count(-1))
    bins = np.arange(np.max(elf_counts) + 1)
    plt.hist(elf_counts, bins=bins)
    plt.xticks(bins)
    plt.ylabel('Counts')
    plt.xlabel('Effective number of labeling functions')
    plt.title(f'Effective number of labeling functions {dataname}')
    # fig.show()
    plt.show()

def per_sample_polarity(dataname, dataset_path, mode='max'):
  train_data, valid_data, test_data = load_dataset(
        dataset_path,
        dataname,
        extract_feature=False
    )
  
  # Get LF polarity
  wl = np.array(train_data.weak_labels)
  n, m = wl.shape
  lf_polarities = []
  for i in range(m):
    s = set(wl[:,i])
    # Don't count abstains
    s.discard(-1)
    print(s)
    lf_polarities.append(len(s))
  
  lf_polarities = np.array(lf_polarities)
  # print(lf_polarities)
  dp_polarities = []
  for dp_wl in wl:
    non_abstains = ~(dp_wl == -1)
    dp_lf_pols = lf_polarities[non_abstains]
    if len(dp_lf_pols) == 0:
      continue  
    
    if mode == 'max':
      dp_pol = np.max(dp_lf_pols)
    elif mode == 'mean':
      dp_pol = np.mean(dp_lf_pols)
    else:
      raise Exception("Mode not valid")
    dp_polarities.append(dp_pol)
  bins = np.linspace(0, np.max(lf_polarities), 30)
  plt.hist(dp_polarities, bins)
  plt.title(f"Per sample polarity for {dataname}")
  plt.xlabel(f'{mode} polarity per sample')
  plt.ylabel('counts')
  plt.show()
    
      

