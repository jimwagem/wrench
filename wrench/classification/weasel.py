import logging
from collections import Counter
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel, BaseLabelModel
from ..config import Config
from ..dataset import BaseDataset
from ..dataset.utils import split_labeled_unlabeled
from ..utils import cross_entropy_with_probs, logit_entropy, norm_reg, acc_reg, bhat_reg
logger = logging.getLogger(__name__)

ABSTAIN = -1


# Source: https://github.com/autonlab/weasel/blob/main/weasel/utils/optimization.py
def mig_loss_function(yhat, output2, p=None):
    # From Max-MIG crowdsourcing paper
    yhat = F.softmax(yhat, dim=1)
    output2 = F.softmax(output2, dim=1)
    batch_size, num_classes = yhat.shape
    I = torch.from_numpy(np.eye(batch_size), )
    E = torch.from_numpy(np.ones((batch_size, batch_size)))
    yhat, output2 = yhat.cpu().float(), output2.cpu().float()
    if p is None:
        p = torch.tensor([1 / num_classes for _ in range(num_classes)]).to(yhat.device)
    new_output = yhat / p
    m = (new_output @ output2.transpose(1, 0))
    noise = torch.rand(1) * 0.0001
    m1 = torch.log(m * I + I * noise + E - I)
    m2 = m * (E - I)
    return -(m1.sum() + batch_size) / batch_size + m2.sum() / (batch_size ** 2 - batch_size)


class Encoder(BackBone):
    def __init__(self, input_size, n_rules, hidden_size, n_class, temperature, dropout=0.3, 
                 balance=None, per_class_acc=True, use_sigmoid=False):
        super(Encoder, self).__init__(n_class=n_class)
        self.use_features = input_size != 0
        self.n_rules = n_rules
        self.acc_scaler = np.sqrt(self.n_rules)
        self.temperature = temperature
        self.per_class_acc = per_class_acc
        self.use_sigmoid = use_sigmoid
        encoder_output_size = n_rules
        if per_class_acc:
            encoder_output_size *= n_class
            self.acc_scaler *= n_class
        self.encoder = nn.Sequential(
            nn.Linear(input_size + n_rules, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, encoder_output_size),
        )
        if balance is None:
            self.log_class_prior = nn.Parameter(torch.log(torch.ones(n_class) / n_class), requires_grad=False)
        else:
            self.log_class_prior = nn.Parameter(torch.log(torch.from_numpy(balance)), requires_grad=False)

    def set_class_prior(self, p):
        for i, pp in enumerate(p):
            self.log_class_prior.data[i] = np.log(p[i])

    def forward(self, batch, return_accuracies=False, use_balance=True):
        device = self.get_device()
        weak_labels = batch['weak_labels'].to(device)
        if self.use_features:
            features = batch['features'].to(device)
            batch_size = features.size(0)

            x = torch.cat((weak_labels, features), 1)
        else:
            batch_size = weak_labels.size(0)
            x = weak_labels

        # Should the accuracies be class dependent
        if self.per_class_acc:
            z = self.encoder(x).view(batch_size, self.n_rules, self.n_class) / self.temperature
        else:
            z = self.encoder(x).view(batch_size, self.n_rules) / self.temperature
            z = torch.unsqueeze(z, dim=2)

        mask = weak_labels != ABSTAIN
        if self.use_sigmoid:
            z = self.acc_scaler * torch.sigmoid(z) * torch.unsqueeze(mask, dim=2)
        else:
            z = self.acc_scaler * torch.softmax(z, dim=1) * torch.unsqueeze(mask, dim=2)

        one_hot = F.one_hot(weak_labels.long() * mask, num_classes=self.n_class)
        z = z * one_hot

        
        if use_balance:
            logits = torch.sum(z, dim=1) + self.log_class_prior
        else:
            logits = torch.sum(z, dim=1)
        
        if return_accuracies:
            return logits, z
        else:
            return logits


class WeaSELModel(BackBone):
    def __init__(self, input_size, n_rules, hidden_size, n_class, temperature,
                 dropout, backbone, balance, loss='ce', use_balance=True, per_class_acc=True, reg_weight=0, use_sigmoid=False):
        super(WeaSELModel, self).__init__(n_class=n_class)
        self.backbone = backbone
        self.encoder = Encoder(input_size, n_rules, hidden_size, n_class, 
                               temperature, dropout, balance, per_class_acc=per_class_acc, use_sigmoid=use_sigmoid)
        self.forward = self.backbone.forward
        self.loss = loss
        self.use_balance=use_balance
        self.per_class_acc = per_class_acc
        self.reg_weight = reg_weight


    def calculate_loss(self, batch, use_hard_labels=False, reg_term=None, c_weights=None):
        predict_f = self.backbone(batch)
        predict_e, z = self.encoder(batch, use_balance=self.use_balance, return_accuracies=True)
        if self.loss == 'ce':
            target_e = torch.softmax(predict_e.detach(), dim=-1)
            target_f = torch.softmax(predict_f.detach(), dim=-1)
            if use_hard_labels:
                target_e = F.one_hot(torch.argmax(target_e, dim=-1), num_classes=self.n_class)
                target_f = F.one_hot(torch.argmax(target_f, dim=-1), num_classes=self.n_class)

            # loss_f_batch = cross_entropy_with_probs(predict_f, target_e, reduction="none") 
            # loss_e_batch = cross_entropy_with_probs(predict_e, target_f, reduction="none")
            
            # # Class weights, currently only for binary
            # if c_weights is not None:
            #     weight = torch.ones_like(loss_f_batch)
            #     weight[predict_e[:,0] <= predict_f[:,0]] *= c_weights[0]
            #     weight[predict_e[:,0] > predict_f[:,0]] *= c_weights[1]
            #     loss_f_batch = loss_f_batch * weight
            # loss_f = loss_f_batch.mean()
            # loss_e = loss_e_batch.mean()
            loss_f = cross_entropy_with_probs(predict_f, target_e) 
            loss_e = cross_entropy_with_probs(predict_e, target_f)
        elif self.loss == 'mig':
            loss_f = mig_loss_function(predict_f, predict_e.detach())
            loss_e = mig_loss_function(predict_e, predict_f.detach())
        else:
            raise ValueError("loss should be 'ce' or 'mig'")

        loss = loss_e + loss_f

        # Calibration regularization
        if reg_term == 'entropy':
            entropy_e = logit_entropy(predict_e)
            entropy_f = logit_entropy(predict_f)
            # print(entropy_e.item(), entropy_f.item())
            
            loss += self.reg_weight*(entropy_e)
        elif reg_term == 'L1':
            d_e = norm_reg(predict_e, p=1, c_weights=c_weights)
            loss += -self.reg_weight*d_e
        elif reg_term == 'L2':
            d_e = norm_reg(predict_e, p=2)
            loss += -self.reg_weight*d_e
        elif reg_term == 'log_acc':
            d_e = acc_reg(z)
            loss += self.reg_weight*d_e
        elif reg_term == 'bhat':
            d_e = bhat_reg(predict_e)
            loss -= self.reg_weight*d_e
        return loss


def init_balance(n_class: int,
                 dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                 y_valid: Optional[np.ndarray] = None
                 ):
    if y_valid is not None:
        y = y_valid
    elif dataset_valid is not None:
        y = np.array(dataset_valid.labels)
    else:
        return np.ones(n_class) / n_class
    class_counts = Counter(y)

    if isinstance(dataset_valid, BaseDataset):
        assert n_class == dataset_valid.n_class

    sorted_counts = np.zeros(n_class)
    for c, cnt in class_counts.items():
        sorted_counts[c] = cnt
    balance = (sorted_counts + 1) / sum(sorted_counts)

    return balance


class WeaSEL(BaseTorchClassModel):
    def __init__(self,
                 temperature: Optional[float] = 1.0,
                 dropout: Optional[float] = 0.3,
                 hidden_size: Optional[int] = 100,

                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 use_balance: Optional[bool] = True,
                 per_class_acc: Optional[bool] = True,
                 reg_weight: Optional[float] = 0,
                 use_sigmoid: Optional[bool] = True, 
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'temperature'     : temperature,
            'dropout'         : dropout,
            'hidden_size'     : hidden_size,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'use_sigmoid'     : use_sigmoid,
            'binary_mode'     : binary_mode,
            'reg_weight'      : reg_weight
        }
        self.model: Optional[WeaSELModel] = None
        self.label_model: Optional[BaseLabelModel] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=False,
            **kwargs
        )
        self.use_balance=use_balance
        self.per_class_acc=per_class_acc
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])

    def fit(self,
            dataset_train: BaseDataset,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            cut_tied: Optional[bool] = False,
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            use_encoder_features: Optional[bool] = True,
            loss: str = 'ce',
            hard_label_step: int = -1,
            reg_term: Optional[str] = None,
            init_model: Optional[bool] = True,
            finalize: Optional[bool] = True,
            c_weights = None,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)
        hyperparas = self.config.hyperparas
        logger.debug(config)

        n_steps = hyperparas['n_steps']
        if hard_label_step == -1:
            hard_label_step = n_steps
        
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        n_rules = dataset_train.n_lf
        n_class = dataset_train.n_class
        balance = init_balance(n_class, dataset_valid=dataset_valid)

        if init_model:
            backbone = self._init_model(
                dataset=dataset_train,
                n_class=dataset_train.n_class,
                config=config,
                is_bert=self.is_bert
            )
            model = WeaSELModel(
                input_size=dataset_train.features.shape[1] if use_encoder_features else 0,
                n_rules=n_rules,
                hidden_size=hyperparas['hidden_size'],
                n_class=n_class,
                temperature=hyperparas['temperature'],
                dropout=hyperparas['dropout'],
                backbone=backbone,
                balance=balance,
                loss=loss,
                use_balance=self.use_balance,
                per_class_acc=self.per_class_acc,
                reg_weight=hyperparas['reg_weight'],
                use_sigmoid=hyperparas['use_sigmoid']
            )
            self.model = model.to(device)
        else:
            model = self.model

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

        labeled_dataset, _ = split_labeled_unlabeled(dataset_train, cut_tied=cut_tied)
        labeled_dataloader = self._init_train_dataloader(
            labeled_dataset,
            n_steps=n_steps,
            config=config,
            return_features=use_encoder_features,
            return_weak_labels=True,
        )

        valid_flag = self._init_valid_step(
            dataset_valid,
            y_valid,
            metric,
            direction,
            patience,
            tolerance,
            return_features=use_encoder_features,
            return_weak_labels=True,
        )

        if valid_flag:
            y = np.array(dataset_valid.labels)
            class_counts = Counter(y)
            sorted_counts = np.zeros(n_class)
            for c, cnt in class_counts.items():
                sorted_counts[c] = cnt
            balance = (sorted_counts + 1) / sum(sorted_counts)
            model.encoder.set_class_prior(balance)

        history = {}
        last_step_log = {}
        use_hard_labels = (hard_label_step == 0)
        try:
            with trange(n_steps, desc="[TRAIN] WeaSEL", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for labeled_batch in labeled_dataloader:
                    loss = model.calculate_loss(labeled_batch, use_hard_labels=use_hard_labels, reg_term=reg_term, c_weights=c_weights)
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break
                        if step >= hard_label_step:
                            use_hard_labels = True

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')
        if finalize:
            self._finalize()

        return history

    def extract_weights(self, dataset):
        dataloader = self._init_valid_dataloader(dataset)
        z_list = []
        for batch in dataloader:
            _, z = self.model.encoder.forward(batch, return_accuracies=True)
            z_list.append(z.detach())
        z_total = torch.cat(z_list, dim=0)
        class_sum = torch.sum(z_total, dim=2)
        means = class_sum.mean(axis=0).numpy()
        return means
