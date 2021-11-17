import json
from pathlib import Path

import numpy as np

from wrench.dataset import load_dataset

data = np.load('professor_teacher_99LFs.npz')  # where * is any file in data/
label_matrix = data['L']  # weak source votes
Xtrain = data['Xtrain']  # features for training on soft labels
Xtest = data['Xtest']  # features for evaluating the model
Ytest = data['Ytest']  # gold labels for evaluating the model

print(Xtrain.shape)
print(Xtest.shape)
print(Ytest.shape)
print(label_matrix.shape)

datasets_path = Path("./wrench/datasets")
profteacher_path = (datasets_path / "profteacher")
profteacher_path.mkdir(exist_ok=True)

# Label mapping
(profteacher_path / "label.json").write_text(json.dumps({"0": "prof", "1": "teacher"}))

train = {}
for idx, (LF, data) in enumerate(zip(label_matrix, Xtrain)):
    train[str(idx)] = {"label": None, "weak_labels": LF.tolist(), "data": {"feature": data.tolist()}}
(profteacher_path / "train.json").write_text(json.dumps(train))

# 250 from test
valid = {}
for idx, (label, data) in enumerate(zip(Ytest[-250:], Xtest[-250:])):
    valid[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": data.tolist()}}
(profteacher_path / "valid.json").write_text(json.dumps(valid))

test = {}
for idx, (label, data) in enumerate(zip(Ytest[:-250], Xtest[:-250])):
    test[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": data.tolist()}}
(profteacher_path / "test.json").write_text(json.dumps(test))

data = load_dataset(datasets_path, dataset="profteacher")
print(data)
