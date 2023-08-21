import os
import numpy as np
import javalang
import torch
from gensim import corpora
from tranverse_tree import _traverse_tree, dfsDict
from sklearn.decomposition import PCA
from parameters import PCA_DIM


def get_flist(dataset, split, cross):
    flist = {}
    if '_' in cross:
        train, test = cross.split('_')
        train_root = f'../dataset/{dataset}/code{train}/'
        test_root = f'../dataset/{dataset}/code{test}/'

        train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
        train_ids = list(train_data.T[0])
        train_ids.extend(train_data.T[1])
        train_ids = set(train_ids)
        all_ids = [int(i.replace('.txt', '')) for i in os.listdir(train_root)]
        test_ids = set(all_ids).difference(train_ids)
        for i in train_ids:
            flist[i] = train_root + str(i) + '.txt'
        for i in test_ids:
            flist[i] = test_root + str(i) + '.txt'
    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        for i in os.listdir(root_path):
            flist[int(i.replace('.txt', ''))] = root_path + i
    fun_ids, paths = [], []
    for i, j in flist.items():
        fun_ids.append(i)
        paths.append(j)
    return fun_ids, paths


def get_source(flist: list):
    print(f'parsing source code: total count {len(flist)}', flush=True)
    source = []
    for path in flist:
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read()
        try:
            tokens = javalang.tokenizer.tokenize(s)
            tree = javalang.parser.parse(tokens)
        except:
            tree = None
        source.append(tree)

    print(f'parse finished, success parsing count {len(source)}', flush=True)
    return source


def get_bow_vec(source_code: list):
    print('build sentences')
    sentences = []
    for tree in source_code:
        sample, size = _traverse_tree(tree)
        listtfinal = []
        dfsDict(sample, listtfinal)
        sentences.append(listtfinal)
    print('build vocab dict')
    vocab_dict = corpora.Dictionary(sentences)
    print('build bow vec')
    vec = []
    for sen in sentences:
        vec.append(vocab_dict.doc2bow(sen))
    vocab_size = len(vocab_dict.dfs)
    print(f'build onr hot vec, vocab:{vocab_size}')
    one_hot = np.zeros(shape=(len(vec), vocab_size), dtype=np.int32)
    for i, v in enumerate(vec):
        for p in v:
            one_hot[i][p[0]] = p[1]
    return one_hot

def getWordEmd(word, listchar,dicttChar):
    listrechar = np.array([0.0 for i in range(0, len(listchar))])
    tt = 1
    for lchar in word:
        listrechar += np.array(((len(word) - tt + 1) * 1.0 / len(word)) * np.array(dicttChar[lchar]))
        tt += 1
    return listrechar

def get_pace_vec(source_code:list):
    print('build sentences')
    sentences = []
    for tree in source_code:
        sample, size = _traverse_tree(tree)
        listtfinal = []
        dfsDict(sample, listtfinal)
        sentences.append(listtfinal)
    print('build vocab dict')

    words = []
    for sen in sentences:
        words.extend(sen)
    words = set(words)
    dicttChar = {}
    def _onehot(i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]
    listchar = ['7', 'I', 'E', 'D', 'u', 'C', 'Y', 'W', 'y', '|', '9', '^', 'X', 't', 'a', 'o', 'Z', 'b', 'A', 'J', 'R',
                'w', '?', 'g', '3', '$', 'B', 'l', '5', 'z', 'v', 'T', '2', 'd', '<', 'e', 'M', 'c', 'S', 'm', '4', 'K',
                'O', 'f', 'i', '=', 'Q', '+', 'x', 'N', '1', 'r', 'p', 'G', 'k', '*', 'q', 'L', 'P', '.', 'n', 'j', 'V',
                'U', '6', '/', '%', '8', 'F', 's', '!', '-', '&', '>', 'h', 'H', '0', '_', '~']
    for i in range(0, len(listchar)):
        dicttChar[listchar[i]] = _onehot(i, len(listchar))
    dictfinalem = {}
    t = 0
    for l in words:
        t += 1
        dictfinalem[l] = getWordEmd(l, listchar, dicttChar)

    all_vec = []
    for s in sentences:
        vec = []
        for w in s:
            vec.append(dictfinalem[w])
        all_vec.append(torch.tensor(vec, dtype=torch.float32))
    return all_vec


def get_pace_torch_data(d, id_map, vec):
    id1, id2 = int(d[0]), int(d[1])
    v1 = vec[id_map[id1]]
    v2 = vec[id_map[id2]]
    return [v1.float(), v2.float()]


def get_torch_data(data, id_map, vec):
    pairs = []
    for d in data:
        id1, id2 = int(d[0]), int(d[1])
        v1 = vec[id_map[id1]]
        v2 = vec[id_map[id2]]
        pairs.append([v1, v2])
    return torch.tensor(pairs, dtype=torch.float32)

def get_pace_train_data(dataset, split, cross):
    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    test_data = np.load(f'../dataset/train_split/{dataset}/test_{split}.npy')
    if os.path.exists(f'./cache/{dataset}_{cross}_allvec.npy'):
        print('reload data from cache')
        fun_ids = np.load(f'./cache/{dataset}_{cross}_allfid.npy', allow_pickle=True)
        all_vec = np.load(f'./cache/{dataset}_{cross}_allvec.npy', allow_pickle=True)
    else:
        print(f'train test data shape: {train_data.shape}, {test_data.shape}', flush=True)
        fun_ids, paths = get_flist(dataset, split, cross)
        source_code = get_source(paths)
        all_vec = get_pace_vec(source_code)
        np.save(f'./cache/{dataset}_{cross}_allfid.npy', fun_ids)
        np.save(f'./cache/{dataset}_{cross}_allvec.npy', all_vec)

    fun_id_map = dict(zip(fun_ids, list(range(len(fun_ids)))))

    return torch.from_numpy(train_data), torch.from_numpy(test_data), all_vec, fun_id_map

def get_train_test_data(dataset, split, cross):
    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    val_data = np.load(f'../dataset/train_split/{dataset}/val_{split}.npy')
    test_data = np.load(f'../dataset/train_split/{dataset}/test_{split}.npy')
    if os.path.exists(f'./cache/{dataset}_{cross}_fid.npy'):
        print('reload data from cache')
        fun_ids = np.load(f'./cache/{dataset}_{cross}_fid.npy')
        pca_vec = np.load(f'./cache/{dataset}_{cross}_vec.npy')
    else:
        print(f'train test data shape: {train_data.shape}, {val_data.shape}, {test_data.shape}', flush=True)
        fun_ids, paths = get_flist(dataset, split, cross)
        source_code = get_source(paths)
        one_hot = get_bow_vec(source_code)
        print(f'one hot vec shape:{one_hot.shape}')
        pca = PCA(n_components=PCA_DIM)
        # pca_vec = pca.fit_transform(one_hot)
        pca_vec = one_hot
        # print(f"pca vec shape:{pca_vec.shape}, pca left info:{np.sum(pca.explained_variance_ratio_)}")
        # del one_hot
        np.save(f'./cache/{dataset}_{cross}_fid.npy', fun_ids)
        np.save(f'./cache/{dataset}_{cross}_vec.npy', pca_vec)

    fun_id_map = dict(zip(fun_ids, list(range(len(fun_ids)))))
    # train_tensor = get_torch_data(train_data, fun_id_map, pca_vec)
    # val_tensor = get_torch_data(val_data, fun_id_map, pca_vec)
    # test_tensor = get_torch_data(test_data, fun_id_map, pca_vec)
    return torch.from_numpy(train_data), torch.from_numpy(val_data), torch.from_numpy(test_data), \
           pca_vec, fun_id_map


if __name__ == '__main__':
    pass
