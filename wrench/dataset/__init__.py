from .basedataset import BaseDataset
from .dataset import NumericDataset, TextDataset, RelationDataset, ImageDataset
from .seqdataset import BaseSeqDataset
from .torchdataset import sample_batch, TorchDataset, BERTTorchTextClassDataset, BERTTorchRelationClassDataset, ImageTorchDataset
import numpy as np

numeric_datasets = ['census', 'basketball', 'tennis', 'commercial', 'imdb_136', 'profteacher', 'amazon', 'crowdsourcing', 'imdb_12']
text_datasets = ['agnews', 'imdb', 'sms', 'trec', 'yelp', 'youtube']
relation_dataset = ['cdr', 'spouse', 'chemprot', 'semeval']
cls_dataset_list = numeric_datasets + text_datasets + relation_dataset
bin_cls_dataset_list = numeric_datasets + ['cdr', 'spouse', 'sms', 'yelp', 'imdb', 'imdb_136', 'youtube', 'amazon', 'profteacher', 'crowdsourcing']
seq_dataset_list = ['laptopreview', 'ontonotes', 'ncbi-disease', 'bc5cdr', 'mit-restaurants', 'mit-movies', 'wikigold', 'conll']

import shutil
from pathlib import Path
import sys
from os import environ, makedirs
from os.path import expanduser, join


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


#### dataset downloading and loading
def get_data_home(data_home=None) -> str:
    data_home = data_home or environ.get('WRENCH_DATA', join('~', 'wrench_data'))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def get_dataset_type(dataset_name):
    if dataset_name in numeric_datasets:
        return NumericDataset
    elif dataset_name in text_datasets:
        return TextDataset
    elif dataset_name in relation_dataset:
        return RelationDataset
    elif dataset_name in seq_dataset_list:
        return BaseSeqDataset
    raise NotImplementedError(f'cannot recognize the dataset type for {dataset_name}! please specify the dataset_type.')


def load_dataset(data_home, dataset, dataset_type=None, extract_feature=False, extract_fn=None, **kwargs):
    if dataset_type is None:
        dataset_class = get_dataset_type(dataset)
    else:
        dataset_class = str_to_class(dataset_type)

    dataset_path = Path(data_home) / dataset
    train_data = dataset_class(path=dataset_path, split='train')
    valid_data = dataset_class(path=dataset_path, split='valid')
    test_data = dataset_class(path=dataset_path, split='test')

    if extract_feature and (dataset_class != BaseSeqDataset):
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)

    return train_data, valid_data, test_data


def load_image_dataset(data_home, dataset, image_root_path, preload_image=True, extract_feature=False, extract_fn='pretrain', **kwargs):
    dataset_path = Path(data_home) / dataset
    train_data = ImageDataset(path=dataset_path, split='train', image_root_path=image_root_path, preload_image=preload_image)
    valid_data = ImageDataset(path=dataset_path, split='valid', image_root_path=image_root_path, preload_image=preload_image)
    test_data = ImageDataset(path=dataset_path, split='test', image_root_path=image_root_path, preload_image=preload_image)

    if extract_feature:
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)

    return train_data, valid_data, test_data

def shuffle_sets(sets, n_lf, n_class):
    weak_labels = np.concatenate([s.weak_labels for s in sets])
    labels = np.concatenate([s.labels for s in sets], axis=0)
    features = np.concatenate([s.features for s in sets], axis=0)
    examples = list(np.concatenate([s.examples for s in sets]))
    splits = [s.split for s in sets]

    lens = [len(s) for s in sets]
    indices = np.arange(sum(lens))
    np.random.shuffle(indices)
    set_indices = []
    counter = 0
    for l in lens:
        set_indices.append(indices[counter:counter+l])
        counter += l
    
    new_datasets = []
    for si, split in zip(set_indices, splits):
        ds = NumericDataset()
        ds.features = np.array([features[i] for i in si])
        ds.labels = [labels[i] for i in si]
        ds.weak_labels = [weak_labels[i] for i in si]
        ds.examples = [examples[i] for i in si]
        ds.n_lf = n_lf
        ds.n_class = n_class
        ds.ids = [str(i) for i in range(len(si))]
        ds.split = split
        new_datasets.append(ds)
    return new_datasets

def has_labels(ds):
    """Determine if the train dataset is mixable, (contains valid labels)"""
    l = ds.labels
    if l is None:
        return False
    if len(l) == 0:
        return False
    if len(np.unique(l)) == 0:
        return False
    return True

def has_weak_labels(ds):
    """Determine if the test dataset is mixable, (contains valid weak labels)"""
    wl = ds.weak_labels
    if wl is None:
        return False
    if len(wl) == 0:
        return False
    if len(wl[0]) == 0:
        return False
    if len(np.unique(wl)) == 1:
        return False
    return True

def resplit_dataset(train, valid, test):
    # weak_labels = np.concatenate((train.weak_labels, valid.weak_labels, test.weak_labels))
    # labels = np.concatenate((train.labels, valid.labels, test.labels), axis=0)
    # features = np.concatenate((train.features, valid.features, test.features), axis=0)
    # examples = train.examples + valid.examples + test.examples
    
    n_class = train.n_class
    n_lf = train.n_lf

    # new_test.split='test'
    if has_labels(train) and has_weak_labels(valid) and has_weak_labels(test):
        print('shuffle all')
        sets = (train, valid, test)
        shuffled = shuffle_sets(sets, n_lf, n_class)
        new_train = shuffled[0]
        new_valid = shuffled[1]
        new_test = shuffled[2]
    elif not has_labels(train) or (not has_weak_labels(valid) and not has_weak_labels(test)):
        print('shuffle test/valid')
        sets = (valid, test)
        shuffled = shuffle_sets(sets, n_lf, n_class)
        new_valid = shuffled[0]
        new_test = shuffled[1]
        new_train = train
    elif has_labels(train) and has_weak_labels(valid):
        print('shuffle train/valid')
        sets = (train, valid)
        shuffled = shuffle_sets(sets, n_lf, n_class)
        new_train = shuffled[0]
        new_valid = shuffled[1]
        new_test = test
    else:
        print('can\'t shuffle')
        new_train, new_valid, new_test = train, valid, test


    return new_train, new_valid, new_test

