'''
[1] Shen Z ,  Cui P ,  Kuang K , et al. Causally Regularized Learning with Agnostic Data Selection Bias[J].  2017.
'''
import os

import torch
from datasets import Dataset


def set_pytorch_seed(seed=824):
    '''
    设置pytorch随机种子

    :param seed:
    :return:
    '''
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def cal_Jbs(X, w, I):
    '''
    计算论文中的Jb，返回包含一个序列，是j=1 -> j=p的 的Jb

    :param X: 输入 n X p
    :param w: 小w n X 1
    :param I: 处理矩阵，表示是否处理，值只有 0， 1 n X p
    :return: Jbs 所有Jb，维度为[p, p]
    '''
    W = w * w
    Jbs = []
    for j in range(X.shape[1]):
        X_minus_j = X.clone().detach()
        X_minus_j[:, j] = 0
        Ij = I[:, j]
        Jb_item = (X_minus_j.T @ (W * Ij) / (W.T @ Ij) - (X_minus_j.T @ (W * (1 - Ij))) / (W.T @ (1 - Ij)))
        Jbs.append(Jb_item.unsqueeze(0))
    return torch.cat(Jbs, dim=0)


def cal_loss_of_confounder_from_Jbs(Jbs):
    '''
    将Jbs求二范数再平方再求和

    :param Jbs:
    :return: 计算结果
    '''
    return (Jbs.norm(2, 1) ** 2).sum()


def cal_partial_Jbs_partial_w(X, w, I, device):
    '''
    计算论文中的 ∂J(ω)/∂ω

    :param X: 输入数据
    :param w: 小w
    :param I: 处理矩阵
    :return: ∂J(ω)/∂ω 列表，包含p个，维度为  [p, p, n]
    '''
    pJpws = []

    the_one = torch.ones(X.shape[1], 1).to(device)
    for j in range(X.shape[1]):
        X_minus_j = X.clone().detach().to(device)
        X_minus_j[:, j] = 0
        Ij = I[:, j].view(1, -1)
        pJpw_item1 = (X_minus_j.T * (the_one @ Ij)) * ((w * w) @ Ij.T) - (X_minus_j.T @ (w * w * w * Ij).T @ Ij)
        pJpw_item1 /= ((w * w) @ Ij.T) ** 2

        Ij = (1 - Ij)
        pJpw_item2 = (X_minus_j.T * (the_one @ Ij)) * ((w * w) @ Ij.T) - (X_minus_j.T @ (w * w * w * Ij).T @ Ij)
        pJpw_item2 /= ((w * w) @ Ij.T) ** 2

        pJpw_item = pJpw_item1 - pJpw_item2
        pJpws.append(pJpw_item)

    return torch.stack(pJpws, dim=0)


def cal_partial_J_parital_w(X, Y, w, I, pred, lambda1=1, lambda2=1, lambda5=1, device='cuda'):
    '''
    计算∂J(ω)/∂ω

    :param X: 输入
    :param Y: 目标输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 这个就是 Xβ
    :return: 对于w的梯度，维度为 n
    '''

    # 计算第一部分
    # 这里是不是可以用神经网络计算出来的loss HMTQ：0422 15：50
    # log_expYX = (1 - 2 * Y) * pred
    # log_expYX = torch.log(1 + torch.exp(log_expYX))
    # part1 = w * w * log_expYX
    # 2022 0425 9:49 改成交叉熵
    part1 = (Y) * torch.log(pred[:, 1]) + (1 - Y) * torch.log(pred[:, 0])
    part1 = -part1
    part1 = w * w * part1
    part1 = part1.sum()

    # 计算第二部分
    the_one = torch.ones(X.shape[1], 1).to(device)
    pJbs_pw = cal_partial_Jbs_partial_w(X, w, I, device)
    Jbs = cal_Jbs(X, w, I)
    part2 = torch.bmm((pJbs_pw * (the_one @ w.view(1, -1))).permute(0, 2, 1), Jbs.view(Jbs.shape[0], -1, 1)).squeeze(-1)
    part2 = part2.sum(0)
    part2 *= lambda1

    # 计算第三部分
    part3 = 4 * lambda2 * w * w * w

    # 计算第四部分
    part4 = w * w - 1
    part4 = part4.sum() ** 2
    part4 = 4 * lambda5 * part4

    # print('1', part1[:2])
    # print('2', part2[:2])
    # print('3', part3[:2])
    # print('4', part4)
    return part1 + part2 + part3 + part4


def cal_Jw(X, Y, w, I, pred, lambda1=1, lambda2=1, lambda5=1, device='cuda'):
    '''
    计算J(w)

    :param X: 输入
    :param Y: 期望输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 预测值 这个就是 Xβ
    :param lambda1:
    :param lambda2:
    :param lambda5:
    :return: 损失， 1 维
    '''
    W = w * w

    # # 2022 0425 9:49 改成交叉熵
    # log_expYX = (1 - 2 * Y) * pred
    # log_expYX = torch.log(1 + torch.exp(log_expYX)).to(device)
    # part1 = W * log_expYX
    # part1 = part1.sum()
    part1 = (Y) * torch.log(pred[:, 1]) + (1 - Y) * torch.log(pred[:, 0])
    part1 = -part1
    part1 = w * w * part1
    part1 = part1.sum()

    part2 = cal_loss_of_confounder_from_Jbs(cal_Jbs(X, w, I))

    part3 = (W.norm(2) ** 2)

    part4 = ((W - 1).sum() ** 2)

    result = part1 + lambda1 * part2 + lambda2 * part3 + lambda5 * part4
    return result, [part1, part2, part3, part4]


def update_w(w, w_g, lr, clip):
    '''
    更新小w

    :param w: 小w
    :param w_g: 小w的梯度
    :param lr: 学习率
    :return: 新w
    '''
    w_g = w_g.clamp(-clip, clip)
    w = w - lr * w_g
    return w


def update_w_one_step(X, Y, w, I, pred, lambda1, lambda2, lambda5, lr, device, clip=50):
    '''
    更新一步，即更新一次w

    :param X: 输入
    :param Y: 期望输出
    :param w: 小w
    :param I: 处理矩阵
    :param pred: 预测值
    :param lambda1: confounder值的超参
    :param lambda2: w的正则化超参
    :param lambda5: 为了让w不为0的超参
    :param lr:
    :return: new_w 新w, loss 当前步的损失
    '''
    w_g = cal_partial_J_parital_w(X, Y, w, I, pred, lambda1=lambda1, lambda2=lambda2, lambda5=lambda5, device=device)
    new_w = update_w(w, w_g, lr, clip)
    loss, loss_detail = cal_Jw(X, Y, w, I, pred, lambda1=lambda1, lambda2=lambda2, lambda5=lambda5, device=device)
    return new_w, loss, loss_detail


def get_aclImdb_sentence_vecs(vec_type, path=None):
    '''
    获得 aclImdb数据级经过Bert编码后的句向量

    :param vec_type: 向量类型 包括['last_hidden_state_first', 'pooler_output']
    :return: train_dataset, test_dataset
    '''

    path = './DataSets/vector' if path is None else path
    train_X = torch.load(os.path.join(path, 'train', f'{vec_type}.pt'))
    train_Y = torch.LongTensor([1] * 12500 + [0] * 12500)
    test_X = torch.load(os.path.join(path, 'test', f'{vec_type}.pt'))
    test_Y = torch.LongTensor([1] * 12500 + [0] * 12500)

    train_dataset = Dataset.from_dict({
        'x': train_X,
        'labels': train_Y
    })

    test_dataset = Dataset.from_dict({
        'x': test_X,
        'labels': test_Y
    })
    return train_dataset, test_dataset


if __name__ == '__main__':
    seed = 824
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    X = torch.randn(100000, 5)
    Y = torch.zeros(X.shape[0])
    Y[torch.rand_like(Y) > 0.5] = 1
    pred = torch.rand(X.shape[0])
    I = torch.zeros_like(X)
    I[torch.rand(*X.shape) > 0.5] = 1
    w = torch.rand(X.shape[0])

    losses = []
    lr = 1e-3
    lr_decay = 0.5
    lambda1 = 1e-2
    lambda2 = 1e-3
    lambda5 = 1e-8
    for i in range(10):
        w, loss, loss_detail = update_w_one_step(X, Y, w, I, pred, lambda1, lambda2, lambda5, lr, device='cpu')
        losses.append(loss)
        lr = lr * lr_decay

    print(losses[:3], losses[-3:])
    print((w * w)[:10])
    print((w * w).sum())
