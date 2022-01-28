import numpy as np
from wrench.synthetic import BaseSyntheticGenerator
from wrench.classification import WeaSEL

class ConstLabel(BaseSyntheticGenerator):
    def __init__(self, use_const_label=True):
        if use_const_label:
            n_lfs = 2
        else:
            n_lfs = 1
        self.use_const_label = use_const_label
        super().__init__(n_class = 2, n_lfs=n_lfs)

    def generate(self, n_data : int=1000):
        ids = list(range(n_data))
        examples = list(range(n_data))
        n_features=4

        data_points = []
        features = []
        weak_labels = []
        labels = []

        for i in range(n_data):
            cluster = np.random.randint(2)
            labels.append(cluster)
            weak_label = [cluster]
            if self.use_const_label:
                weak_label.append(0)
            if cluster == 0:
                feature = np.random.normal(np.zeros(n_features), 1).astype(np.float32)
            elif cluster == 1:
                feature = np.random.normal(np.ones(n_features), 1).astype(np.float32)
            weak_labels.append(weak_label)
            features.append(feature)
        
        return {
            'ids'        : ids,
            'examples'   : examples,
            'labels'     : labels,
            'weak_labels': weak_labels,
            'features'   : np.array(features)
        }

if __name__ == '__main__':
    cl = ConstLabel(use_const_label=True)
    train_data = cl.generate_split(split='train', n_data=1000)
    valid_data = cl.generate_split(split='valid', n_data=500)
    test_data = cl.generate_split(split='test', n_data=500)
    model = WeaSEL(
            temperature=1.0,
            dropout=0.3,
            hidden_size=100,
            batch_size=64,
            real_batch_size=8,
            test_batch_size=128,
            n_steps=10000,
            grad_norm=1.0,
            backbone='MLP',
            # backbone='BERT',
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0,
        )
    model.fit(
        train_data,
        valid_data,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device='cpu',
        verbose=True
    )
    test_result = model.test(test_data, 'acc')
    print(test_result)
    predictions = model.predict(test_data)
    for a, b in zip(predictions, test_data.labels):
        print(a, b)

    


