import logging
import torch
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
from wrench.classification import WeaSEL
from snorkel.utils import probs_to_preds
import sklearn.metrics as cls_metrics
from wrench.utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from wrench.labelmodel import Snorkel
from wrench.endmodel import EndClassifierModel
from wrench.synthetic.dataset_generator import SubSampledLFDataset

def max_metric(probs, true_labels, metric, num=100, plot=False):
    p = probs[:,0]
    max_p = max(p)
    min_p = min(p)
    c_range = np.linspace(min_p, max_p, num)


    metric_vals = []
    for c in c_range:
        preds = np.zeros_like(p)
        preds[p < c] = 1
        metric_vals.append(metric(true_labels, preds))
    c_max_index = np.argmax(metric_vals)
    c_max = c_range[c_max_index]
    print(f'Max cutoff {c_max}')

    if plot:
        plt.plot(c_range, metric_vals)
        plt.title('Metric value vs cutoff')
        plt.grid()
        plt.show()
    return metric_vals[c_max_index]




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cpu')

# Seed 2 with no split had really bad regular performance
# set_seed(1)
#### Load dataset
dataset_path = '../../datasets/'
data = 'imdb_136'
# bert_model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert',
    cache_name='bert',
    device=device,
)
train_data = SubSampledLFDataset(train_data)
# train_data, valid_data, test_data = resplit_dataset(train_data, valid_data, test_data)
#### Run WeaSEL
model = WeaSEL(
    temperature=1.0,
    dropout=0.3,
    hidden_size=70,

    batch_size=128,
    real_batch_size=64,
    test_batch_size=64,
    n_steps=20000,
    grad_norm=1.0,

    backbone='FlexMLP',
    # backbone='MLP',
    # backbone='BERT',
    backbone_model_name='MLP',
    backbone_fine_tune_layers=-1,  # fine  tune all
    optimizer='Adam',
    optimizer_lr=5e-5,
    optimizer_weight_decay=7e-7,
    use_balance=False,
    per_class_acc=False,
    use_sigmoid=True,
    reg_weight=0.2
)
model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='logloss',
    patience=200,
    device=device,
    reg_term='L1'
)
metric = model.test(test_data, 'acc')
logger.info(f'WeaSEL testacc: {metric}')
metric = model.test(test_data, 'f1_binary')
logger.info(f'WeaSEL testf1: {metric}')
logger.info(f'max f1: {max_metric(model.predict_proba(test_data), test_data.labels, cls_metrics.matthews_corrcoef, plot=True)}')
# ece = model.test(test_data, 'ece')
# logger.info(f'WeaSEL test ece: {ece}')




