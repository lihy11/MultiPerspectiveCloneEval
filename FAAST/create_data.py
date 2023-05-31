import os
import numpy as np
import shutil


def write_train_data(dataset, split):
    for l in ['train', 'test', 'val']:
        from_path = f'../dataset/train_split/{dataset}/{l}_{split}.npy'
        data = np.load(from_path)
        with open(f'./{dataset}/{l}_{split}.txt', 'w') as f:
            for d in data:
                f.write(f'{d[0]}.txt {d[1]}.txt {d[2]}\n')


def copy_code(dataset, split, cross):
    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    train_ids = list(train_data.T[0])
    train_ids.extend(train_data.T[1])
    train_ids = list(set(train_ids))
    all_ids = [int(i.replace('.txt', '')) for i in os.listdir(f'../dataset/{dataset}/code0/')]
    test_ids = set(all_ids).difference(train_ids)

    train, test = cross.split('_')
    for i in train_ids:
        shutil.copy(f'../dataset/{dataset}/code{train}/{i}.txt', f'./{cross}/{i}.txt')
    for i in test_ids:
        shutil.copy(f'../dataset/{dataset}/code{test}/{i}.txt', f'./{cross}/{i}.txt')


def get_train_data(dataset: str, split: str, cross: str):
    """
    :param cross: default: None, or like {i}_{j}, train on i, test on j
    :param dataset: [gcj , cbcb , cross_gcj , cross_cbcb]
    :param split: [random{i}, fun{i}, pro{i}, cross_{i}_{j}]
    :return: None
    """

    train_data_path = f'./{dataset}/train_{split}.txt'
    if not os.path.exists(train_data_path):
        write_train_data(dataset, split)

    # create a new dataset dir for cross test
    if '_' in cross:
        if not os.path.exists(f'./{cross}'):
            os.mkdir(f'./{cross}')
        copy_code(dataset, split, cross)


def get_flist(dataset, split, cross):
    flist = []
    if '12' == cross or '21' == cross:
        root_path = f'../dataset/{dataset}/codecomp/'
        flist.extend([root_path + i for i in os.listdir(root_path)])
    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        flist.extend([root_path + i for i in os.listdir(root_path)])

    return flist


def get_train_test_data(dataset, split, cross):
    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    val_data = np.load(f'../dataset/train_split/{dataset}/val_{split}.npy')
    test_data = np.load(f'../dataset/train_split/{dataset}/test_{split}.npy')

    if '12' == cross:
        train_data = np.load(f'../dataset/train_split/{dataset}/old.npy')
        test_data = np.load(f'../dataset/train_split/{dataset}/new.npy')
        root_path = f'../dataset/{dataset}/codecomp/'
        train_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in train_data]
        test_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in test_data]
        return train_data, None, test_data
    elif '21' == cross:
        train_data = np.load(f'../dataset/train_split/{dataset}/new.npy')
        test_data = np.load(f'../dataset/train_split/{dataset}/old.npy')
        root_path = f'../dataset/{dataset}/codecomp/'
        train_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in train_data]
        test_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in test_data]
        return train_data, None, test_data
    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        train_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in train_data]
        val_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in val_data]
        test_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in test_data]
        return train_data, val_data, test_data
