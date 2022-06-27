import logging
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import MajorityVoting
from wrench.endmodel import EndClassifierModel
import sklearn.metrics as cls_metrics
from sklearn.calibration import calibration_curve

if __name__=="__main__":
    device = torch.device('cpu')
    dataset_path = '../../datasets/'
    data = 'imdb_136'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert',
        cache_name='bert',
        device=device,
    )
    label_model = MajorityVoting()
    model = EndClassifierModel(
        batch_size=128,
        test_batch_size=512,
        n_steps=10000,
        backbone='FlexMLP',
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7,
    )
    label_model.fit(train_data)
    train_data = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)
    labels = test_data.labels

    # Two modes
    use_mv = True
    if use_mv:
        train_labels = aggregated_hard_labels
    else:
        train_labels = labels
    model.fit(
        dataset_train=train_data,
        y_train=train_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='auc',
        patience=100,
        device=device
    )

    probs = model.predict_proba(test_data)
    prob_true, prob_pred = calibration_curve(labels, probs[:,1], n_bins=30)
    plt.plot(prob_pred, prob_true)
    x_range = np.linspace(0,1,100)
    plt.plot(x_range, x_range)
    plt.ylim(0,1)
    plt.title('Calibration curve')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    if use_mv:
        save_path = './mv_calcurve.png'
    else:
        save_path = './gt_calcurve.png'
    plt.savefig(save_path)
    plt.show()
