import json
from pathlib import Path

import numpy as np

from wrench.dataset import load_dataset

if __name__ == "__main__":
    dataset_name = "crowdsourcing"
    dtype = "TextDataset"
    # see: http://labelme2.csail.mit.edu/Release3.0/browserTools/php/publications.php
    labels = {"0": "Objects", "1": "Cars", "2": "Person", "3": "Building", "4": "Road", "5": "Sidewalk", "6": "Sky", "7": "Tree"}

    # source: https://drive.google.com/drive/folders/1BBHeiLIr3txUCs5tX9pL5JCN5S8JEzEA
    data_path = Path("prepared")
    data = {
        name.stem: np.load(str(name)) for name in data_path.glob("*.npy")
    }

    for name, value in data.items():
        print(name, value.shape)

    datasets_path = Path("../datasets")
    dataset_path = (datasets_path / dataset_name)
    dataset_path.mkdir(exist_ok=True)

    # Label mapping
    (dataset_path / "label.json").write_text(json.dumps(labels))

    train = {}
    for idx, (label, LF, feature) in enumerate(zip(data["labels_train"], data['answers'], data['data_train_vgg16'])):
        train[str(idx)] = {"label": label.tolist(), "weak_labels": LF.tolist(), "data": {"feature": feature.tolist()}}
    (dataset_path / "train.json").write_text(json.dumps(train))

    # 250 from test
    valid = {}
    for idx, (label, feature) in enumerate(zip(data['labels_valid'], data['data_valid_vgg16'])):
        valid[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
    (dataset_path / "valid.json").write_text(json.dumps(valid))

    test = {}
    for idx, (label, feature) in enumerate(zip(data['labels_test'], data['data_test_vgg16'])):
        test[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
    (dataset_path / "test.json").write_text(json.dumps(test))

    datasets = load_dataset(datasets_path, dataset=dataset_name, dataset_type=dtype)
    print(datasets)
