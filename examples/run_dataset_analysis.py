import numpy as np
from wrench.dataset_analysis import determine_balance, majority_proportion, \
    lf_maj_proportion, effective_lf_count, per_sample_polarity


if __name__=="__main__":
    path = '../../datasets'
    datasets = ['profteacher', 'imdb', 'imdb_136', 'amazon', 'crowdsourcing']
    # per_sample_polarity('spouse', path, mode='mean')
    for ds in datasets:
        effective_lf_count(ds, path)
        per_sample_polarity(ds, path, mode='mean')