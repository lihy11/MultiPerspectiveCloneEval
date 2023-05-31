import argparse
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
from create_data import get_flist
from createclone import createast, creategmndata, createseparategraph

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--dataset", default='cbcb_new')
parser.add_argument("--split", default='random0')
parser.add_argument("--cross", type=str, default='0')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=True)
parser.add_argument("--ifedge", default=True)
parser.add_argument("--whileedge", default=True)
parser.add_argument("--foredge", default=True)
parser.add_argument("--blockedge", default=True)
parser.add_argument("--nexttoken", default=True)
parser.add_argument("--nextuse", default=True)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=20)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()

dataset = args.dataset
split = args.split
cross = args.cross
result_path = f'./{dataset}_{cross}_result/'
model_path = f'./{dataset}_{cross}_models/'
if not os.path.exists(result_path):
    os.mkdir(result_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)

device = torch.device(f'cuda:{args.cuda}')
# device = torch.device('cuda:{args.cuda}')
astdict, vocablen, vocabdict = createast(get_flist(dataset=dataset, split=split, cross=cross))

treedict = createseparategraph(astdict, vocablen, vocabdict, device, mode=args.graphmode, nextsib=args.nextsib,
                               ifedge=args.ifedge, whileedge=args.whileedge, foredge=args.foredge,
                               blockedge=args.blockedge, nexttoken=args.nexttoken, nextuse=args.nextuse)
traindata, validdata, testdata = creategmndata(dataset, split, cross, treedict, vocablen, vocabdict, device)

num_layers = int(args.num_layers)
model = models.GMNnet(vocablen, embedding_dim=100, num_layers=num_layers, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CosineEmbeddingLoss()
criterion2 = nn.MSELoss()


def create_batches(data):
    # random.shuffle(data)
    batches = [data[graph:graph + args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches


def test(dataset):
    # model.eval()
    count = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        x1 = torch.tensor(x1, dtype=torch.long, device=device)
        x2 = torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1 != None:
            edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction = model(data)
        output = F.cosine_similarity(prediction[0], prediction[1])
        results.append(output.item())
        prediction = torch.sign(output).item()

        if prediction > args.threshold and label.item() == 1:
            tp += 1
            # print('tp')
        if prediction <= args.threshold and label.item() == -1:
            tn += 1
            # print('tn')
        if prediction > args.threshold and label.item() == -1:
            fp += 1
            # print('fp')
        if prediction <= args.threshold and label.item() == 1:
            fn += 1
            # print('fn')
    print(f"tp: {tp}, tn: {tn},fp: {fp}, fn: {fn}")
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print('precision is none')
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print('recall is none')
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print(f'precision: {p}, recall: {r}, F1: {f1}')
    return results


for epoch in range(args.num_epochs):  # without batching
    print(epoch, flush=True)
    batches = create_batches(traindata)
    totalloss = 0.0
    main_index = 0.0
    for index, batch in enumerate(batches):
        optimizer.zero_grad()
        batchloss = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)
            # print(len(data))
            # for i in range(len(data)):
            # print(i)
            # data[i]=torch.tensor(data[i], dtype=torch.long, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 != None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            prediction = model(data)
            # batchloss=batchloss+criterion(prediction[0],prediction[1],label)
            cossim = F.cosine_similarity(prediction[0], prediction[1])
            batchloss = batchloss + criterion2(cossim, label)
            # print(cossim.shape, label.shape)
        batchloss.backward()
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        if index % 1000 == 0:
            print("Epoch (Loss=%g)" % round(loss, 5), flush=True)

    if validdata != None:
        print('eval result:', flush=True)
        devresults = test(validdata)
        devfile = open(result_path + split + '_val_' + str(epoch + 1), mode='w')
        for res in devresults:
            devfile.write(str(res) + '\n')
        devfile.close()

    print('test result:', flush=True)
    testresults = test(testdata)
    resfile = open(result_path + split + '_test_' + str(epoch + 1), mode='w')
    for res in testresults:
        resfile.write(str(res) + '\n')
    resfile.close()

    torch.save(model, model_path + split + str(epoch + 1))
