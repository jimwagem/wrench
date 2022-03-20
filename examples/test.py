import logging
import torch
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import Snorkel, MajorityVoting
from wrench.endmodel import EndClassifierModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cpu')

#### Load dataset
dataset_path = '../../datasets/'
data = 'amazon'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert', # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)
label_model = MajorityVoting()

label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
acc = label_model.test(train_data, 'acc')
logger.info(f'label model train acc: {acc}')

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)

#### Run end model: MLP
model = EndClassifierModel(
    batch_size=64,
    test_batch_size=64,
    n_steps=10000,
    backbone='FlexMLP',
    optimizer='Adam',
    optimizer_lr=5e-5,
    optimizer_weight_decay=7e-7,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_hard_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='auc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (MLP) test acc: {acc}')
