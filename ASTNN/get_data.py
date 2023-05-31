import numpy as np
import javalang
import pandas as pd
from gensim.models.word2vec import Word2Vec
import os


def get_flist(dataset, split, cross):
    flist = {}
    if '12' == cross or '21'==cross:
        root_path = f'../dataset/{dataset}/codecomp/'
        for i in os.listdir(root_path):
            flist[int(i.replace('.txt', ''))] = root_path + i

    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        for i in os.listdir(root_path):
            flist[int(i.replace('.txt', ''))] = root_path + i
    return flist


def parse_source(flist: dict):
    print(f'parsing source code: total count {len(flist)}', flush=True)
    source = {'id': [], 'code': []}
    for fun_id, path in flist.items():
        source['id'].append(fun_id)
        with open(path, 'r') as f:
            s = f.read()
        try:
            tokens = javalang.tokenizer.tokenize(s)
            tree = javalang.parser.parse(tokens)
        except:
            tree = None
        source['code'].append(tree)

    source = pd.DataFrame(source)
    source = source.dropna()
    print(f'parse finished, success parsing count {source.shape[0]}', flush=True)
    return source


def dictionary_and_embedding(source_df, size):
    from utils import get_sequence as func
    def trans_to_sequences(ast):
        sequence = []
        func(ast, sequence)
        return sequence

    corpus = source_df['code'].apply(trans_to_sequences)
    s = set([])
    for i in corpus:
        for j in i:
            s.add(j)
    print(len(s))
    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
    return w2v


def generate_block_seqs(source_df, w2v):
    from utils import get_blocks_v1 as func

    word2vec = w2v.wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        func(r, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree

    trees = pd.DataFrame(source_df, copy=True)
    codes = []
    for _, row in trees.iterrows():
        # print(row['id'])
        # if row['id'] != 6933:
        #     continue
        codes.append(trans2seq(row['code']))
    trees['code'] = codes
    if 'label' in trees.columns:
        trees.drop('label', axis=1, inplace=True)
    return trees


def merge(data_np, blocks):
    pairs = pd.DataFrame(data_np, columns=['id1', 'id2', 'label'])
    pairs.loc[pairs['label'] == -1, 'label'] = 0
    pairs['id1'] = pairs['id1'].astype(int)
    pairs['id2'] = pairs['id2'].astype(int)
    df = pd.merge(pairs, blocks, how='left', left_on='id1', right_on='id')
    df = pd.merge(df, blocks, how='left', left_on='id2', right_on='id')
    df.drop(['id_x', 'id_y'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

def get_comp_data(dataset, cross):
    flist = get_flist(dataset, None, cross)
    source = parse_source(flist)
    w2v = dictionary_and_embedding(source, 128)
    trees = generate_block_seqs(source, w2v)

    print(f'trees size:{trees.shape}, vocab_size:{w2v.wv.syn0.shape[0]}')
    trees.to_pickle(f'./cache/{dataset}_{cross}_trees.pkl')
    w2v.save(f'./cache/{dataset}_{cross}_w2v')
    if cross == '12':
        train_data = np.load(f'../dataset/train_split/{dataset}/old.npy')
        test_data = np.load(f'../dataset/train_split/{dataset}/new.npy')
    elif cross == '21':
        train_data = np.load(f'../dataset/train_split/{dataset}/new.npy')
        test_data = np.load(f'../dataset/train_split/{dataset}/old.npy')

    train_data = merge(train_data, trees)
    test_data = merge(test_data, trees)
    return train_data, test_data, w2v

def get_train_test_data(dataset, split, cross):

    if os.path.exists(f'./cache/{dataset}_{cross}_trees.pkl'):
        trees = pd.read_pickle(f'./cache/{dataset}_{cross}_trees.pkl')
        w2v = Word2Vec.load(f'./cache/{dataset}_{cross}_w2v')
        print(f'reload data from cache, trees size:{trees.shape}, vocab_size:{w2v.wv.syn0.shape[0]}')
    else:
        flist = get_flist(dataset=dataset, split=split, cross=cross)
        source = parse_source(flist)
        w2v = dictionary_and_embedding(source, 128)
        trees = generate_block_seqs(source, w2v)

        print(f'trees size:{trees.shape}, vocab_size:{w2v.wv.syn0.shape[0]}')
        trees.to_pickle(f'./cache/{dataset}_{cross}_trees.pkl')
        w2v.save(f'./cache/{dataset}_{cross}_w2v')

    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    test_data = np.load(f'../dataset/train_split/{dataset}/test_{split}.npy')

    train_data = merge(train_data, trees)
    test_data = merge(test_data, trees)
    return train_data, test_data, w2v
