import json
from pathlib import Path

import numpy as np

from wrench.dataset import load_dataset

data = np.load('professor_teacher_99LFs.npz')
for idx in data.files:
    print(idx, data[idx].shape)

dataset_name = "profteacher"
datasets_path = Path("../datasets")
profteacher_path = (datasets_path / dataset_name)
profteacher_path.mkdir(exist_ok=True)

# Label mapping
(profteacher_path / "label.json").write_text(json.dumps({"0": "prof", "1": "teacher"}))

train = {}
for idx, (label, LF, feature) in enumerate(zip(data['Ytrain_gold'], data['L'], data['Xtrain'])):
    train[str(idx)] = {"label": label.tolist(), "weak_labels": LF.tolist(), "data": {"feature": feature.tolist()}}
(profteacher_path / "train.json").write_text(json.dumps(train))

# 250 from test
valid = {}
for idx, (label, feature) in enumerate(zip(data['Ytest'][-250:], data['Xtest'][-250:])):
    valid[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
(profteacher_path / "valid.json").write_text(json.dumps(valid))

test = {}
for idx, (label, feature) in enumerate(zip(data['Ytest'][:-250], data['Xtest'][:-250])):
    test[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
(profteacher_path / "test.json").write_text(json.dumps(test))

datasets = load_dataset(datasets_path, dataset=dataset_name)
print(datasets)
