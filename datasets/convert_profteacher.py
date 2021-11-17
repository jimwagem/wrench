import json
from pathlib import Path

import numpy as np

from wrench.dataset import load_dataset


# source: https://drive.google.com/drive/folders/1v7IzA3Ab5zDEsRpLBWmJnXo5841tSOlh
ds = [
    ('IMDB_12LFs.npz', "imdb_12", {"0": "prof", "1": "teacher"}),
    ('IMDB_136LFs.npz', "imdb_136", {"0": "prof", "1": "teacher"}),
    ('professor_teacher99LFs.npz', "profteacher", {"0": "prof", "1": "teacher"}),
    ('Amazon_175LFs.npz', "amazon", {"0": "prof", "1": "teacher"}),
]


for file_name, dataset_name, labels in ds:
    data = np.load(file_name)
    for idx in data.files:
        print(idx, data[idx].shape)

    datasets_path = Path("../datasets")
    dataset_path = (datasets_path / dataset_name)
    dataset_path.mkdir(exist_ok=True)

    # Label mapping
    (dataset_path / "label.json").write_text(json.dumps(labels))

    train = {}
    for idx, (label, LF, feature) in enumerate(zip(data['Ytrain_gold'], data['L'], data['Xtrain'])):
        train[str(idx)] = {"label": label.tolist(), "weak_labels": LF.tolist(), "data": {"feature": feature.tolist()}}
    (dataset_path / "train.json").write_text(json.dumps(train))

    # 250 from test
    valid = {}
    for idx, (label, feature) in enumerate(zip(data['Ytest'][-250:], data['Xtest'][-250:])):
        valid[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
    (dataset_path / "valid.json").write_text(json.dumps(valid))

    test = {}
    for idx, (label, feature) in enumerate(zip(data['Ytest'][:-250], data['Xtest'][:-250])):
        test[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
    (dataset_path / "test.json").write_text(json.dumps(test))

    datasets = load_dataset(datasets_path, dataset=dataset_name)
    print(datasets)
