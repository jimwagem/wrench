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

show=False
def max_mcc(probs, true_labels, reg, num=100, plot=False, save=False, data='test'):
    p = probs[:,0]
    max_p = max(p)
    min_p = min(p)
    c_range = np.linspace(min_p, max_p, num)[1:]

    metric = cls_metrics.matthews_corrcoef
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
        plt.title('MCC as a function of prediction cutoff')
        plt.xlabel('Cutoff value')
        plt.ylabel('MCC')
        plt.grid()
        if save:
            plt.savefig(f'./{data}/max_mcc_{reg}.pdf')
        if show:
            plt.show()
        else:
            plt.clf()
    return metric_vals[c_max_index]

def prediction_hist(probs, reg, save=False, data='test'):
    p = probs[:,0]
    plt.hist(p, bins=100)
    plt.xlabel('Prediction $P(Y=0)$')
    plt.ylabel('count')
    plt.title('Histogram of prediction scores')
    if save:
        plt.savefig(f'./{data}/predictions_{reg}.pdf')
    if show:
        plt.show()
    else:
        plt.clf()

def weight_histogram(weights, weak_labels, save=False, correct_coverage=False, data='test'):
    # Only works on datasets with polarity 1
    n_lf = len(weights)
    index = np.arange(n_lf)

    lf_classes = np.zeros(n_lf)
    for i, wl in enumerate(np.transpose(weak_labels)):
        if 1 in np.unique(wl):
            lf_classes[i] = 1
    zero_index = index[lf_classes == 0]
    one_index = index[lf_classes == 1]
    print(f'num 0 lfs: {len(zero_index)}')
    print(f'num 1 lfs: {len(one_index)}')

    if correct_coverage:
        lfa = LFAnalysis(np.array(weak_labels))
        scale_fac = lfa.lf_coverages()
        print(f'Total class 0 lf_cov: {np.sum(scale_fac[zero_index])}')
        print(f'Total class 1 lf_cov: {np.sum(scale_fac[one_index])}')
    else:
        scale_fac = np.ones(len(weights))
    weights = weights / scale_fac
    plt.bar(zero_index, weights[zero_index], width=0.8, color='orange', label='class 0')
    plt.bar(one_index, weights[one_index], width=0.8, color='blue', label='class 1')
    plt.legend()
    plt.xlabel('LF number')
    plt.ylabel('LF weight')
    if save:
        plt.savefig(f'./{data}/weights_{reg}.pdf')
    if show:
        plt.show()
    else:
        plt.clf()

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
    correct_lf_balance = True
    if correct_lf_balance:
        data += '_balanced'
        train_data = SubSampledLFDataset(train_data)
    save=True
    # for reg in ['None', 'L1']:
    with open('./reg_weight_dict.pkl', 'rb') as f:
        reg2weight = pickle.load(f)
    for reg in ['None','L1','bhat','entropy','log_acc']:
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
            finalize=True
        )
        labels = test_data.labels
        probs = model.predict_proba(test_data)
        max_f1 = max_mcc(probs, labels, reg=reg, plot=True, save=save, data=data)
        prediction_hist(probs, reg=reg, save=save, data=data)

        # # Calibration curve
        n_bins = 30
        if reg == 'None':
            n_bins= 100
        prob_true, prob_pred = calibration_curve(labels, probs[:,1], n_bins=n_bins)
        plt.scatter(prob_pred, prob_true)
        plt.title('Calibration curve')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        if save:
            plt.savefig(f'./{data}/calibration_curve_{reg}.pdf')
        if show:
            plt.show()
        else:
            plt.clf()
        weights = model.extract_weights(train_data)
        weight_histogram(weights, train_data.weak_labels, save=save, correct_coverage=True, data=data)


