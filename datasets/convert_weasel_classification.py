import json
from pathlib import Path

import numpy as np

from wrench.dataset import load_dataset


# source: https://drive.google.com/drive/folders/1v7IzA3Ab5zDEsRpLBWmJnXo5841tSOlh
ds = [
    ('IMDB_136LFs.npz', "imdb_136", {"0": "Negative", "1": "Positive"}, "TextDataset"),
    ('professor_teacher_99LFs.npz', "profteacher", {"0": "prof", "1": "teacher"}, "TextDataset"),
    ('Amazon_175LFs.npz', "amazon", {"0": "Negative", "1": "Positive"}, "TextDataset"),
    ('IMDB_12LFs.npz', "imdb_12", {"0": "Negative", "1": "Positive"}, "TextDataset")
]
# ds = [
#     ('IMDB_12LFs.npz', "imdb_12", {"0": "Negative", "1": "Positive"}, "TextDataset"),
#     ('IMDB_136LFs.npz', "imdb_136", {"0": "Negative", "1": "Positive"}, "TextDataset")
# ]

if __name__ == "__main__":
    for file_name, dataset_name, labels, dtype in ds:
        data = np.load(file_name)
        print(f"dataset={dataset_name}")
        for idx in data.files:
            print(idx, data[idx].shape)

        gold = None
        if dataset_name == "imdb_136":
            # IMDB136 misses Ytrain_gold
            data2 = np.load("IMDB_12LFs.npz")
            gold = data2["Ytrain_gold"]

            # Only label matrix L is different
            assert (data['Ytest'] == data2['Ytest']).all()
            assert (data['Xtrain'] == data2['Xtrain']).all()
            assert (data['Xtest'] == data2['Xtest']).all()

        if gold is None:
            gold = data["Ytrain_gold"]

        datasets_path = Path("../../datasets")
        dataset_path = (datasets_path / dataset_name)
        dataset_path.mkdir(exist_ok=True)

        # Label mapping
        (dataset_path / "label.json").write_text(json.dumps(labels))

        train = {}
        for idx, (label, LF, feature) in enumerate(zip(gold, data['L'], data['Xtrain'])):
            train[str(idx)] = {"label": label.tolist(), "weak_labels": LF.tolist(), "data": {"feature": feature.tolist()}}
        (dataset_path / "train.json").write_text(json.dumps(train))

        # 250 from test
        # Place in valid when splitval[i]==0
        total_test_valid = len(data['Ytest'])
        assert (total_test_valid > 250)
        splitval = np.ones(total_test_valid)
        splitval[:250] = 0
        np.random.shuffle(splitval)


        valid = {}
        test = {}
        valid_counter = 0
        test_counter = 0
        for idx, (label, feature) in enumerate(zip(data['Ytest'], data['Xtest'])):
            if splitval[idx] == 0:
                dataset = valid
                counter = valid_counter
                valid_counter += 1
            else:
                dataset = test
                counter = test_counter
                test_counter += 1
            dataset[str(counter)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}
        (dataset_path / "valid.json").write_text(json.dumps(valid))
        (dataset_path / "test.json").write_text(json.dumps(test))

        # for idx, (label, feature) in enumerate(zip(data['Ytest'][:-250], data['Xtest'][:-250])):
        #     test[str(idx)] = {"label": label.tolist(), "weak_labels": [], "data": {"feature": feature.tolist()}}

        datasets = load_dataset(datasets_path, dataset=dataset_name, dataset_type=dtype)
        print(datasets)
