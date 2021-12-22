from typing import Optional, Union

import numpy as np

from .syntheticdataset import BaseSyntheticGenerator

class GaussianLF():
    def __init__(self, means, feature_mask):
        self.means = means
        self.feature_mask = feature_mask
        self.n_class = len(means)
        

    def predict(self, data_point):
        min_diff = np.inf
        best_label = None
        for index in range(self.n_class):
            diff_vec = self.means[index] - data_point
            diff = np.linalg.norm(diff_vec*self.feature_mask)
            if diff < min_diff:
                min_diff = diff
                best_label = index
        return best_label
    
class MultiGaussian(BaseSyntheticGenerator):
    def __init__(self,
                n_features: int,
                n_visible_features: int,
                n_good_lfs: int,
                n_class: int,
                n_bad_lfs: int = 0,
                n_random_lfs: int = 0,
                n_constant_lfs: int = 0,
                sample_low: int = 0,
                sample_high: int = 1):
        super().__init__(n_class=n_class, n_lfs=n_good_lfs + n_bad_lfs + n_random_lfs + n_constant_lfs)
        self.n_features = n_features
        self.n_visible_features = n_visible_features
        self.n_class = n_class
        self.n_good_lfs = n_good_lfs
        self.n_bad_lfs = n_bad_lfs
        self.n_constant_lfs = n_constant_lfs
        self.n_random_lfs = n_random_lfs
        self.mus = []
        for i in range(n_class):
            mu = (sample_high - sample_low) * np.random.random(size=n_features) + sample_low
            self.mus.append(mu)
        
        self.fake_mus = []
        for i in range(n_class):
            mu = (sample_high - sample_low) * np.random.random(size=n_features) + sample_low
            self.fake_mus.append(mu)

        # make labeling functions
        self.good_lfs = []
        self.bad_lfs = []
        for i in range(self.n_good_lfs):
            indices = np.random.choice(self.n_class, self.n_visible_features)
            mask = np.zeros(self.n_features)
            mask[indices] = 1
            self.good_lfs.append(GaussianLF(self.mus, mask))
        
        for i in range(self.n_bad_lfs):
            indices = np.random.choice(n_class, self.n_visible_features)
            mask = np.zeros(self.n_features)
            mask[indices] = 1
            self.bad_lfs.append(GaussianLF(self.fake_mus, mask))
        
    
    def generate(self, n_data: int = 1000):
        ids = list(range(n_data))
        examples = list(range(n_data))

        data_points = []
        labels = []
        fake_labels = []
        for i in range(n_data):
            label = np.random.randint(self.n_class)
            fake_label = np.random.randint(self.n_class)
            labels.append(label)
            fake_labels.append(fake_label)
            # std 1
            dp_real = np.random.normal(self.mus[label], 1).astype(np.float32)
            dp_fake = np.random.normal(self.fake_mus[fake_label], 1).astype(np.float32)
            data_points.append((dp_real, dp_fake))

        

        weak_labels = []
        for dp_real, dp_fake in data_points:
            constant = np.random.randint(self.n_class)
            dp_weak_labels = []
            for lf in self.good_lfs:
                dp_weak_labels.append(lf.predict(dp_real))
            for lf in self.bad_lfs:
                dp_weak_labels.append(lf.predict(dp_fake))    
            for _ in range(self.n_random_lfs):
                dp_weak_labels.append(np.random.randint(self.n_class))
            for _ in range(self.n_constant_lfs):
                dp_weak_labels.append(constant)
            weak_labels.append(dp_weak_labels)
        
        features = np.array([np.concatenate(x) for x in data_points])

        return {
            'ids'        : ids,
            'examples'   : examples,
            'labels'     : labels,
            'weak_labels': weak_labels,
            'features'   : features
        }

            

class ConditionalIndependentGenerator(BaseSyntheticGenerator):
    def __init__(self,
                 n_class: int,
                 n_lfs: int,
                 class_prior: Optional[Union[list, np.ndarray]] = None,
                 lf_prior: Optional[Union[list, np.ndarray]] = None,
                 alpha: Optional[float] = 0.7,
                 alpha_radius: Optional[float] = 0.5,
                 beta: Optional[float] = 0.1,
                 beta_radius: Optional[float] = 0.1,
                 random_state=None):
        super().__init__(n_class, n_lfs, class_prior, lf_prior, random_state)
        self.alpha_l = self.generator.uniform(low=max(0, alpha - alpha_radius), high=min(1, alpha + alpha_radius), size=n_lfs)
        self.beta_l = self.generator.uniform(low=max(0, beta - beta_radius), high=min(1, beta + beta_radius), size=n_lfs)

    def generate(self, n_data: int = 1000):
        ids = list(range(n_data))
        examples = list(range(n_data))
        labels = list(self.generator.choice(self.n_class, size=n_data, p=self.class_prior))
        weak_labels = []
        for i, y in enumerate(labels):
            weak_label = []
            for alpha, beta, target in zip(self.alpha_l, self.beta_l, self.lf_targets):
                if target == y:
                    p = alpha * beta / self.class_prior[y]
                else:
                    p = (1 - alpha) * beta / (self.class_prior[y] * (self.n_class - 1))
                if self.generator.random() < p:
                    weak_label.append(target)
                else:
                    weak_label.append(-1)
            weak_labels.append(weak_label)
        return {
            'ids'        : ids,
            'examples'   : examples,
            'labels'     : labels,
            'weak_labels': weak_labels,
        }


class DataDependentGenerator(ConditionalIndependentGenerator):
    def __init__(self,
                 n_class: int,
                 n_lfs: int,
                 n_cluster: int = 10,
                 n_cluster_per_lfs: int = 2,
                 class_prior: Optional[Union[list, np.ndarray]] = None,
                 lf_prior: Optional[Union[list, np.ndarray]] = None,
                 alpha: Optional[float] = 0.7,
                 beta: Optional[float] = 0.1,
                 gamma: Optional[float] = 0.3,
                 alpha_radius: Optional[float] = 0.5,
                 random_state=None):
        super().__init__(n_class, n_lfs, class_prior, lf_prior, alpha, beta, alpha_radius, random_state)
        self.n_cluster = n_cluster
        self.n_cluster_per_lfs = n_cluster_per_lfs
        self.lf_pro_clusters = [self.generator.choice(n_cluster, size=n_cluster_per_lfs) for _ in range(n_lfs)]
        self.gamma_l = self.generator.uniform(low=0, high=gamma, size=n_lfs)

    def generate(self, n_data: int = 1000):
        ids = list(range(n_data))
        examples = list(range(n_data))
        labels = list(self.generator.choice(self.n_class, size=n_data, p=self.class_prior))
        clusters = list(self.generator.choice(self.n_cluster, size=n_data))
        weak_labels = []
        for i, y in enumerate(labels):
            weak_label = []
            cluster = clusters[i]
            for alpha, beta, gamma, target, lf_pro_clusters in \
                    zip(self.alpha_l, self.beta_l, self.gamma_l, self.lf_targets, self.lf_pro_clusters):
                if cluster not in lf_pro_clusters:
                    alpha = max(alpha - gamma, 0.1)
                if target == y:
                    p = alpha * beta / self.class_prior[y]
                else:
                    p = (1 - alpha) * beta / (self.class_prior[y] * (self.n_class - 1))
                if self.generator.random() < p:
                    weak_label.append(target)
                else:
                    weak_label.append(-1)
            weak_labels.append(weak_label)
        return {
            'ids'        : ids,
            'examples'   : examples,
            'labels'     : labels,
            'weak_labels': weak_labels,
        }


class CorrelatedGenerator(ConditionalIndependentGenerator):
    def __init__(self,
                 n_class: int,
                 n_lfs: int,
                 n_overlap: Optional[int] = 0,
                 n_conflict: Optional[int] = 0,
                 n_duplicate: Optional[int] = 1,
                 class_prior: Optional[Union[list, np.ndarray]] = None,
                 lf_prior: Optional[Union[list, np.ndarray]] = None,
                 alpha: Optional[float] = 0.7,
                 beta: Optional[float] = 0.1,
                 alpha_radius: Optional[float] = 0.3,
                 overlap_theta: Optional[float] = 0.8,
                 conflict_theta: Optional[float] = 0.8,
                 random_state=None):
        self.n_overlap = n_overlap
        self.overlap_theta = overlap_theta
        self.n_conflict = n_conflict
        self.conflict_theta = conflict_theta
        self.n_duplicate = n_duplicate
        assert n_overlap + n_conflict + n_duplicate < n_lfs
        super().__init__(n_class, n_lfs, class_prior, lf_prior, alpha, beta, alpha_radius, random_state)

        lf_pool = list(range(self.n_lfs))
        self.overlap_lfs = self.generator.choice(lf_pool, size=n_overlap, replace=False)
        lf_pool = [i for i in lf_pool if i not in self.overlap_lfs]
        self.conflict_lfs = self.generator.choice(lf_pool, size=n_conflict, replace=False)
        lf_pool = [i for i in lf_pool if i not in self.conflict_lfs]
        self.duplicate_lfs = self.generator.choice(lf_pool, size=n_duplicate, replace=False)
        self.normal_lfs = [i for i in lf_pool if i not in self.duplicate_lfs]
        self.overlap_target_lf = self.generator.choice(self.normal_lfs, size=n_overlap, replace=False)
        self.conflict_target_lf = self.generator.choice(self.normal_lfs, size=n_conflict, replace=False)
        self.duplicate_target_lf = self.generator.choice(self.normal_lfs, size=n_duplicate, replace=False)
        self.dep_graph = []
        for overlap_lf, target in zip(self.overlap_lfs, self.overlap_target_lf):
            self.lf_targets[overlap_lf] = self.lf_targets[target]
            self.dep_graph.append((overlap_lf, target))
        for conflict_lf, target in zip(self.conflict_lfs, self.conflict_target_lf):
            self.lf_targets[conflict_lf] = self.sampel_other_label(self.lf_targets[target])
            self.dep_graph.append((conflict_lf, target))
        for duplicate_lf, target in zip(self.duplicate_lfs, self.duplicate_target_lf):
            self.lf_targets[duplicate_lf] = self.lf_targets[target]
            self.dep_graph.append((duplicate_lf, target))

    def generate(self, n_data: int = 1000):
        ids = list(range(n_data))
        examples = list(range(n_data))
        labels = list(self.generator.choice(self.n_class, size=n_data, p=self.class_prior))
        weak_labels = []
        for i, y in enumerate(labels):
            weak_label = -np.ones(self.n_lfs, dtype=int)

            for j in self.normal_lfs:
                alpha, beta, target = self.alpha_l[j], self.beta_l[j], self.lf_targets[j]
                if target == y:
                    p = alpha * beta / self.class_prior[y]
                else:
                    p = (1 - alpha) * beta / (self.class_prior[y] * (self.n_class - 1))
                if self.generator.random() < p:
                    weak_label[j] = target

            for j, m in zip(self.overlap_lfs, self.overlap_target_lf):
                target = self.lf_targets[j]
                if weak_label[m] != -1:
                    if self.generator.random() < self.overlap_theta:
                        weak_label[j] = target
                else:
                    alpha, beta = self.alpha_l[j], self.beta_l[j]
                    if target == y:
                        p = alpha * beta / self.class_prior[y]
                    else:
                        p = (1 - alpha) * beta / (self.class_prior[y] * (self.n_class - 1))
                    if self.generator.random() < p:
                        weak_label[j] = target

            for j, m in zip(self.conflict_lfs, self.conflict_target_lf):
                target = self.lf_targets[j]
                if weak_label[m] != -1:
                    if self.generator.random() < self.conflict_theta:
                        weak_label[j] = target
                else:
                    alpha, beta = self.alpha_l[j], self.beta_l[j]
                    if target == y:
                        p = alpha * beta / self.class_prior[y]
                    else:
                        p = (1 - alpha) * beta / (self.class_prior[y] * (self.n_class - 1))
                    if self.generator.random() < p:
                        weak_label[j] = target

            for j, m in zip(self.duplicate_lfs, self.duplicate_target_lf):
                weak_label[j] = weak_label[m]

            weak_labels.append(weak_label.tolist())
        return {
            'ids'        : ids,
            'examples'   : examples,
            'labels'     : labels,
            'weak_labels': weak_labels,
        }
