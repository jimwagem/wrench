import numpy as np
from wrench.dataset_analysis import determine_balance, majority_proportion, \
    lf_maj_proportion, effective_lf_count, per_sample_polarity


if __name__=="__main__":
    datasets = ['profteacher', 'imdb', 'imdb_136', 'amazon', 'crowdsourcing']
    per_sample_polarity('crowdsourcing', '../../datasets', mode='mean')