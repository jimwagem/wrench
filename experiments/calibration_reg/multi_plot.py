import logging
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
from snorkel.labeling import LFAnalysis
import sklearn.metrics as cls_metrics
from sklearn.calibration import calibration_curve
from wrench.utils import set_seed
from wrench.classification import WeaSEL
from wrench.evaluation import mcc
from wrench.synthetic.dataset_generator import SubSampledLFDataset

show=True

if __name__=="__main__":
    device = torch.device('cpu')

    dataset_path = '../../../datasets/'
    data = 'amazon'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert',
        cache_name='bert',
        device=device,
    )
    correct_lf_balance = False
    if correct_lf_balance:
        data += '_balanced'
        train_data = SubSampledLFDataset(train_data)
    save=True
    with open('./reg_weight_dict.pkl', 'rb') as f:
        reg2weight = pickle.load(f)
    # for reg in ['None','L1','bhat','entropy','log_acc']:
    reg = 'L1'
    for seed in range(5):
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
          reg_weight=reg2weight[reg],
          use_sigmoid=True
      )
      wl = np.array(train_data.weak_labels)
      n_zero_vote = np.sum(wl == 0)
      n_one_vote = np.sum(wl == 1)
      total = n_zero_vote + n_one_vote
      zero_frac = n_zero_vote/total
      one_frac = n_one_vote/total
      metric = 'auc' if reg == 'None' else 'logloss'
      model.fit(
          dataset_train=train_data,
          dataset_valid=valid_data,
          evaluation_step=10,
          metric=metric,
          patience=200,
          device=device,
          hard_label_step=-1,
          init_model=True,
          reg_term=reg,
          finalize=True,
          c_weights=[zero_frac, one_frac]
      )
      labels = test_data.labels
      probs = model.predict_proba(test_data)

      # # Calibration curve
      n_bins = 30
      if reg == 'None':
          n_bins= 100
      prob_true, prob_pred = calibration_curve(labels, probs[:,1], n_bins=n_bins)
      plt.plot(prob_pred, prob_true, alpha=0.7, color='blue')
    plt.title('Calibration curve')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    x_range=np.linspace(0,1,100)
    plt.plot(x_range, x_range, color='red', ls='--')
    if save:
        plt.savefig(f'./{data}_calibration_curve_{reg}_weighted_multi.pdf')
    if show:
        plt.show()
    else:
        plt.clf()
