import torch
import numpy as np
import pandas as pd
import json





from sklearn.model_selection import train_test_split



import trust_model

device ='cuda'

def permute(array):
    if array.dtype == torch.float32:
        return array.permute(1, 0, 2)
    elif array.dtype == torch.int or array.dtype == torch.long:
        return array.permute(1, 0)
    else:
        raise AssertionError

def train(model, optimizer, criterion, dataloader, device='cuda'):
    model.train()
    loss_sum = 0
    correct = list()

    num_sample = 0
    for (src, mask, target_index), label in dataloader:
        src = {key: permute(value).to(device) for key, value in src.items()}
        mask = {'{}_mask'.format(key): value.to(device) for key, value in mask.items()}
        label = label.to(device)

        target_index = target_index.to(device)

        optimizer.zero_grad()
        out = model(**src, **mask)
        batch_size = out.shape[1]

        y = out[target_index, torch.arange(batch_size, device=device)]

        loss = criterion(y[label != -100], label[label != -100])


        loss.backward()
        optimizer.step()

        num_valid = (label != -100).sum()
        num_sample += num_valid
        loss_sum += loss.detach() * num_valid

    report = dict(loss=(loss_sum / num_sample).to('cpu').item())
    return report


def test(model, criterion, dataloader, device=device):
    model.eval()
    loss_sum = 0
    correct = 0
    m = torch.nn.Sigmoid()
    num_sample = 0
    for (src, mask, target_index), label in dataloader:
        src = {key: permute(value).to(device) for key, value in src.items()}
        mask = {'{}_mask'.format(key): value.to(device) for key, value in mask.items()}
        label = label.to(device)

        with torch.no_grad():
            out = model(**src, **mask)
            batch_size = out.shape[1]
            y = out[target_index, torch.arange(batch_size)]

            loss = criterion(y[label != -100], label[label != -100])

            num_valid = (label != -100).sum()
            num_sample += num_valid
            loss_sum += loss.detach() * num_valid
            correct += (y.argmax(axis=1)[label != -100] == label[label != -100]).sum()

    report = dict(loss=(loss_sum / num_sample).item(), correct=(correct/num_sample).detach().to('cpu').item())
    return report


def main(batch_size=32, num_iter=50, n_head=8, n_feature=128, n_hidden=128, n_out=11, n_layers=3, lr=0.00005, out_dir='results', orig_model_path=None, target=[0,1,2,3,4], dropout=0.5):
    import dataset
    user_ids_train, user_ids_test = list(zip(*dataset.users_valid.groupby('mode').apply(lambda grp: train_test_split(grp.sort_index().index, random_state=42, test_size=0.2)).values))

    dataset_train, dataset_test = dataset.Dataset(np.hstack(user_ids_train), with_trust=True, n=2), dataset.Dataset(np.hstack(user_ids_test), with_trust=True)
    dataset_train, dataset_test = map(tuple, [dataset_train, dataset_test])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    encoder = trust_model.MyTransformerEncoder(n_head=n_head, n_feature=n_feature, dropout=dropout, n_hidden=n_hidden, n_layers=n_layers)
    decoder = trust_model.MyTransformerDecoder(n_out=n_out, n_in=n_feature, target=target, dropout=dropout, n_hidden=n_hidden, n_head=n_head)
    model = trust_model.PredictionModel2(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    if orig_model_path is not None:
        model.load_state_dict(torch.load(orig_model_path))
    model = model.to(device)


    train_results = list()
    test_results = list()
    max_loss = 800000
    for i in range(num_iter):
        train_results.append(train(model, optimizer, criterion, dataloader_train))
        test_results.append(test(model, criterion, dataloader_test))
        loss = test_results[-1]['loss']
        print(i, train_results[-1], test_results[-1], end=' ')
        if loss < max_loss:
            print('(new record)')
            model.to('cpu')
            torch.save(model.state_dict(), out_dir+'/{}.pt'.format(i))
            model.to('cuda')
            max_loss = loss
        else:
            print()
        with open(out_dir+'/log.json', 'w') as f:
            json.dump(dict(train=train_results, test=test_results), f)

    df_train_results = pd.DataFrame(train_results)
    df_test_results = pd.DataFrame(test_results)
    df_train_results['type'] = 'train'
    df_test_results['type'] = 'test'
    results = pd.concat([df_train_results, df_test_results]).reset_index().pivot(index='index', columns='type', values=['loss'])

    return results

if __name__ == '__main__':
    main(num_iter=300, n_head=8, n_layers=3, lr=2e-5)
