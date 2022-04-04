import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
import sklearn.metrics as cls_metrics
from sklearn.calibration import calibration_curve
from wrench.utils import set_seed
from wrench.classification import WeaSEL
from wrench.evaluation import mcc

def max_mcc(probs, true_labels, reg, num=100, plot=False, save=False):
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
            plt.savefig(f'./max_mcc_{reg}.pdf')
        plt.show()
    return metric_vals[c_max_index]

def prediction_hist(probs, reg, save=False):
    p = probs[:,0]
    plt.hist(p, bins=100)
    plt.xlabel('Prediction $P(Y=0)$')
    plt.ylabel('count')
    plt.title('Histogram of prediction scores')
    if save:
        plt.savefig(f'./predictions_{reg}.pdf')
    plt.show()

if __name__=="__main__":
    device = torch.device('cpu')

    dataset_path = '../../../datasets/'
    data = 'profteacher'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert',
        cache_name='bert',
        device=device,
    )
    save=False
    for reg in ['L1']:
        # set_seed(1)
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
            per_class_acc=False
        )
        metric = 'auc' if reg == 'None' else 'logloss'
        # model.fit(
        #     dataset_train=train_data,
        #     dataset_valid=valid_data,
        #     evaluation_step=10,
        #     metric='auc',
        #     patience=200,
        #     device=device,
        #     hard_label_step=-1,
        #     init_model=True,
        #     reg_term=None
        # )
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
            finalize=False
        )
        labels = test_data.labels
        probs = model.predict_proba(test_data)
        max_f1 = max_mcc(probs, labels, reg=reg, plot=True, save=save)
        prediction_hist(probs, reg=reg, save=save)

        # Calibration curve
        n_bins = 1000 if reg == 'None' else 30
        prob_true, prob_pred = calibration_curve(labels, probs[:,1], n_bins=n_bins)
        plt.scatter(prob_pred, prob_true)
        plt.title('Calibration curve')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        if save:
            plt.savefig(f'./calibration_curve_{reg}.pdf')
        plt.show()


