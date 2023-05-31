import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from model import LinearModel
from get_data import get_train_test_data, get_torch_data
from parameters import PCA_DIM, BATCH_SIZE, LR


def val_result(model, dataloader, y_true, device, id_map, vec):
    val_pred = []
    for i, (x, y) in enumerate(dataloader):
        x = get_torch_data(x, id_map, vec)
        x, y = x.to(device), y.to(device)
        out1, out2 = model(x[:, 0, :], x[:, 1, :])
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

    train_data, val_data, test_data, vec, fun_id_map = get_train_test_data(dataset, split, cross)

    device = torch.device(device_name)
    model = LinearModel(vec.shape[1])
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss(reduction='mean')

    train_dataset = TensorDataset(train_data[:, 0:2], train_data[:, 2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data[:, 0:2], val_data[:, 2]), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_data[:, 0:2], test_data[:, 2]), batch_size=batch_size)

    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(train_loader):
            x = get_torch_data(x, fun_id_map, vec)
            x, y = x.to(device), y.to(device).float()
            optim.zero_grad()
            out1, out2 = model(x[:, 0, :], x[:, 1, :])
            dis = torch.cosine_similarity(out1, out2)
            loss = lossfn(dis, y)

            loss.backward()
            optim.step()

            if i % 500 == 0:
                print(i, '-----', loss.item())

        p, r, f1 = val_result(model, val_loader, val_data[:, 2], device, fun_id_map, vec)
        print(f'epoch : {epoch}, val result: [{p}, {r}, {f1}]', flush=True)
        p, r, f1 = val_result(model, test_loader, test_data[:, 2], device, fun_id_map, vec)
        print(f'epoch : {epoch}, test result: [{p}, {r}, {f1}]', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a dataset:[gcj|cbcb]")
    parser.add_argument('--dataset', default='gcj')
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
