import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from model import PaceModel
from get_data import get_pace_train_data, get_pace_torch_data
from parameters import PCA_DIM, BATCH_SIZE, LR


def val_result(model, testdata, y_true, device, id_map, vec):
    val_pred = []
    for i, x in enumerate(testdata):
        x = get_pace_torch_data(x, id_map, vec)
        x = [x[0].to(device), x[1].to(device)]
        out1, out2 = model(x[0], x[1])
        dis = torch.cosine_similarity(out1, out2)
        val_pred.append(dis.cpu().detach())
    val_pred = torch.cat(val_pred)
    val_pred[val_pred >= 0] = 1
    val_pred[val_pred < 0] = -1
    val_pred = val_pred.int()
    p, r, f1, _ = precision_recall_fscore_support(y_true, val_pred, average='binary')
    return p, r, f1


def train(dataset, split, cross, device_name):
    print('training: ----', flush=True)
    EPOCHS = 50
    lr = LR
    batch_size = BATCH_SIZE

    train_data, test_data, all_vec, fun_id_map = get_pace_train_data(dataset, split, cross)

    device = torch.device(device_name)
    model = PaceModel(all_vec[0].shape[1])
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss(reduction='mean')

    train_dataset = TensorDataset(train_data[:, 0:2], train_data[:, 2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(EPOCHS):
        for i, (X, Y) in enumerate(train_loader):
            Y = Y.to(device).float()
            out = []
            for j in range(X.shape[0]):
                x = get_pace_torch_data(X[j], fun_id_map, all_vec)
                x = [x[0].to(device), x[1].to(device)]
                out1, out2 = model(x[0], x[1])
                dis = torch.cosine_similarity(out1, out2)
                out.append(dis)
            out = torch.vstack(out)
            optim.zero_grad()
            loss = lossfn(out, Y)
            loss.backward()
            optim.step()

            if i % 500 == 0:
                print(i, '-----', loss.item())

        p, r, f1 = val_result(model, test_data, test_data[:, 2], device, fun_id_map, all_vec)
        print(f'epoch : {epoch}, test result: [{p}, {r}, {f1}]', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a dataset:[gcj|cbcb]")
    parser.add_argument('--dataset', default='cbcb')
    parser.add_argument('--split', default='fun0')
    parser.add_argument('--cross', default='1')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    cross = args.cross
    cuda = args.cuda
    if torch.cuda.is_available():
        device_name = f'cuda:{cuda}'
    else:
        device_name = 'cpu'

    train(dataset, split, cross, device_name)
