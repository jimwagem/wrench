import logging
import torch
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
from wrench.classification import WeaSEL
import sklearn.metrics as cls_metrics
import numpy as np

def max_f1(probs, true_labels, num=1000):
    p = probs[:,0]
    max_p = max(p)
    min_p = min(p)
    c_range = np.linspace(min_p, max_p, num)

    f1s = []
    for c in c_range:
        preds = np.zeros_like(p)
        preds[p < c] = 1
        f1s.append(cls_metrics.f1_score(true_labels, preds))
    print(np.max(f1s))




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cpu')

#### Load dataset
dataset_path = '../../datasets/'
data = 'youtube'
# bert_model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert',
    cache_name='bert',
    device=device,
)

train_data, valid_data, test_data = resplit_dataset(train_data, valid_data, test_data)
#### Run WeaSEL
model = WeaSEL(
    temperature=1.0,
    dropout=0.3,
    hidden_size=70,

    batch_size=64,
    real_batch_size=64,
    test_batch_size=64,
    n_steps=10000,
    grad_norm=1.0,

    backbone='MLP',
    # backbone='BERT',
    backbone_model_name='MLP',
    backbone_fine_tune_layers=-1,  # fine  tune all
    optimizer='AdamW',
    optimizer_lr=1e-4,
    optimizer_weight_decay=7e-7
)
model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_binary',
    patience=100,
    device=device
)
# f1 = model.test(test_data, 'f1_binary')
# probs = model.predict_proba(test_data)
# true_labels = test_data.labels
# max_f1(probs, true_labels)
# logger.info(f'WeaSEL test f1: {f1}')

# f1 = model.test(test_data, 'f1_binary')
# logger.info(f'WeaSEL test f1: {f1}')
ece = model.test(test_data, 'ece')
logger.info(f'WeaSEL test ece: {ece}')




