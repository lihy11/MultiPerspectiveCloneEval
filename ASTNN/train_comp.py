import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support

from get_data import get_comp_data

warnings.filterwarnings('ignore')


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[gcj|cbcb]")
    parser.add_argument('--dataset', default='gcj')
    parser.add_argument('--cross', default='12')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    dataset = args.dataset
    cross = args.cross

    print("Train for ", str.upper(dataset + '  ' + cross))
    train_data, test_data, w2v = get_comp_data(dataset=dataset,  cross=cross)

    word2vec = w2v.wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 20
    BATCH_SIZE = 16
    USE_GPU = True

    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    precision, recall, f1 = 0, 0, 0
    print('Start training...', flush=True)

    train_data_t, test_data_t = train_data, test_data
    # training procedure
    for epoch in range(EPOCHS):
        start_time = time.time()
        # training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data_t):
            batch = get_batch(train_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            train1_inputs, train2_inputs, train_labels = batch
            if USE_GPU:
                train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train1_inputs, train2_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()
            if i % 5000 == 0:
                print(i, loss.item(), '\n', flush=True)
        print("Testing-%d..." % epoch)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test1_inputs, test2_inputs)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        print("Total testing results(P,R,F1):%.3f, %.3f, %.3f\n" % (precision, recall, f1))
