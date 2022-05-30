import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter

from utils.models import Vector_Classifier
from utils.utils import set_pytorch_seed, update_w_one_step, get_aclImdb_sentence_vecs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")


def eval(model, data_loader):
    model.eval()

    m = [0] * 4
    m_items = []
    loss = None
    for d in data_loader:
        x = torch.stack(d['x'], dim=1).float().to(device)
        x_indexes = d['x_indexes'].to(device)
        labels = d['labels'].to(device)
        results = model(x=x, x_indexes=x_indexes, labels=labels, mode='eval')

        loss = results['loss']
        prediction = results['prediction']
        m_items += (labels * 2 + prediction * 1).tolist()

    unique, count = np.unique(m_items, return_counts=True)
    data_count = dict(zip(unique, count))

    for k in data_count.keys():
        m[k] += data_count[k]

    p = m[3] / (m[3] + m[1] + 1)
    r = m[3] / (m[3] + m[2] + 1)
    f1 = 2 * (p * r) / (p + r)
    acc = (m[0] + m[3]) / sum(m)
    _00_number = m[0]
    _01_number = m[1]
    _10_number = m[2]
    _11_number = m[3]
    return {
        'p': p,
        'r': r,
        'f1': f1,
        'acc': acc,
        'detail/_00_number': _00_number,
        'detail/_01_number': _01_number,
        'detail/_10_number': _10_number,
        'detail/_11_number': _11_number,
        'loss': loss
    }


def train(model, data_loader, optimizer):
    model.train()
    loss_item = 0
    n_total = 0

    predictions = []
    pred_orign = []
    loss_detial = torch.zeros(3)

    for d in data_loader:
        x = torch.stack(d['x'], dim=1).float().to(device)
        x_indexes = d['x_indexes'].to(device)
        labels = d['labels'].to(device)
        results = model(x=x, x_indexes=x_indexes, labels=labels)

        predictions += results['prediction'].tolist()
        pred_orign += results['pred_orign'].tolist()
        loss = results['loss']
        loss_detial = results['loss_detail']

        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_item += loss.item()
        n_total += len(d)

    return loss_item / n_total, predictions, pred_orign, (loss_detial / n_total).tolist()


def start_train_normal(batch_size=64, lr_model=2e-4, epochs=1000, log_name='crl_normal'):
    writer = SummaryWriter('./logs/' + log_name)
    model = Vector_Classifier(hidden_size=768, num_labels=2).to(device)

    train_dataset, eval_dataset = get_aclImdb_sentence_vecs('last_hidden_state_first')

    train_dataset, eval_dataset = handle_dataset(train_dataset, eval_dataset)

    train_dataset = train_dataset.add_column('x_indexes', list(range(len(train_dataset))))
    eval_dataset = eval_dataset.add_column('x_indexes', list(range(len(eval_dataset))))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    ####
    # 设置超参
    w = torch.ones(len(train_dataset)).to(device)
    model.set_causal_hyperparameter(w, lambdas)
    #
    ###

    optimizer = AdamW(model.parameters(), lr=lr_model)

    for i in trange(epochs):
        loss1, pred, pred_orign, loss1_detial = train(model, train_dataloader, optimizer)

        results = eval(model, eval_dataloader)
        for k in results.keys():
            writer.add_scalar(f'eval/{k}', results[k], global_step=i)
        writer.add_scalar('loss/model', loss1, global_step=i)
        writer.add_scalar('loss/model_examples_loss', loss1_detial[0], global_step=i)
        writer.add_scalar('loss/model_L2_loss', loss1_detial[1], global_step=i)
        writer.add_scalar('loss/model_L1_loss', loss1_detial[2], global_step=i)

    writer.flush()
    writer.close()


def start_train_reweight(batch_size=64, lr_model=1e-3, lr_w=1e-4, epochs=1000, log_name='crl_causal',
                         change_distribution=False):
    writer = SummaryWriter('./logs/' + log_name)
    model = Vector_Classifier(hidden_size=768, num_labels=2).to(device)

    train_dataset, eval_dataset = get_aclImdb_sentence_vecs('last_hidden_state_first')

    train_dataset, eval_dataset = handle_dataset(train_dataset, eval_dataset)

    train_dataset = train_dataset.add_column('x_indexes', list(range(len(train_dataset))))
    eval_dataset = eval_dataset.add_column('x_indexes', list(range(len(eval_dataset))))

    ####
    # 设置超参
    Y = torch.tensor(train_dataset['labels']).to(device)
    X = train_dataset['x']

    X = torch.stack([torch.tensor(x) for x in X], dim=0).float().to(device)
    w = torch.randn(len(X)).to(device)

    model.set_causal_hyperparameter(w, lambdas)
    I = torch.zeros_like(X).to(device)
    I[X > X.mean(0)] = 1

    ##  更改X的分布
    if change_distribution:
        X[I == 1] = 1
        X[I == 0] = 0
        train_dataset = train_dataset[:]
        train_dataset['x'] = X
        train_dataset = Dataset.from_dict(train_dataset)

    ###
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr_model)

    for i in trange(epochs):
        loss1, pred, pred_orign, loss1_detail = train(model, train_dataloader, optimizer)
        pred_orign = torch.tensor(pred_orign).to(device)

        new_w, loss2, loss2_detail = update_w_one_step(X, Y, model.causal_hyper['w'], I, pred_orign, lambdas[1],
                                                       lambdas[2],
                                                       lambdas[5], lr_w, device=device)
        model.causal_hyper['w'] = new_w

        results = eval(model, eval_dataloader)
        for k in results.keys():
            writer.add_scalar(f'eval/{k}', results[k], global_step=i)
        writer.add_scalar('loss/model', loss1, global_step=i)
        writer.add_scalar('loss/model_examples_loss', loss1_detail[0], global_step=i)
        writer.add_scalar('loss/model_L2_loss', loss1_detail[1], global_step=i)
        writer.add_scalar('loss/model_L1_loss', loss1_detail[2], global_step=i)

        writer.add_scalar('loss/weight', loss2.item(), global_step=i)
        writer.add_scalar('loss/weight_examples_loss', loss2_detail[0].item(), global_step=i)
        writer.add_scalar('loss/weight_confounder_loss', loss2_detail[1].item(), global_step=i)
        writer.add_scalar('loss/weight_L2_loss', loss2_detail[2].item(), global_step=i)
        writer.add_scalar('loss/weight_avoid_0_loss', loss2_detail[3].item(), global_step=i)

    writer.add_text('loss/weight', str(model.causal_hyper['w'] ** 2))

    writer.flush()
    writer.close()


def handle_dataset(train_dataset, eval_dataset):
    new_traind = {}
    new_testd = {}
    for k in train_dataset[:1]:
        new_traind[k] = train_dataset[:train_num_pos][k] + train_dataset[-train_num_neg:][k]
        new_testd[k] = eval_dataset[:eval_num][k] + eval_dataset[-eval_num:][k]
    train_dataset = Dataset.from_dict(new_traind)
    eval_dataset = Dataset.from_dict(new_testd)
    return train_dataset, eval_dataset


###########超参
# 正例个数
train_num_pos = 18
# 负例个数
train_num_neg = 2
# 测试集中的正负例个数，测试集总数为 2 * eval_num
eval_num = 1000
# 一些超参，跟论文公式中对应
lambdas = {
    1: 1e-3,# confounder loss
    2: 1e-2,# w l2
    3: 1e-5,# f l2
    4: 1e-5,# f l1
    5: 1e-1,# avoid 0
}
# 随机种子
set_pytorch_seed(626)
# 批大小
batch_size = 100
# 批数
epochs = 10
# 模型学习率
lr = 1e-2
# w学习率
lr_w = 1e-3
# 是否干预分布
change_distribution = True
# 梯度裁剪
clip=None
#####超参


if __name__ == '__main__':
    start_train_reweight(epochs=epochs,
                         lr_model=lr, lr_w=lr_w, batch_size=batch_size, change_distribution=change_distribution)
    start_train_normal(epochs=epochs,
                       lr_model=lr, batch_size=batch_size)
