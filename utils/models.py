import torch
import torch.nn as nn


class Vector_Classifier(nn.Module):
    causal_hyper = None

    def __init__(self, hidden_size, num_labels):
        super(Vector_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_labels)
        )

        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, x, x_indexes, labels=None, mode='train'):
        x = self.classifier(x)
        prediction = torch.softmax(x, dim=1)

        if mode == 'train':
            if Vector_Classifier.causal_hyper is None:
                raise ValueError('请设置超参')

            w = Vector_Classifier.causal_hyper['w']
            lambda3 = Vector_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_Classifier.causal_hyper['lambdas'][4]

            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()

            W = w * w

            W_selected = torch.gather(W, 0, x_indexes)

            # log_expYX = (1 - labels) * prediction[:, 0] + (labels) * prediction[:, 1]
            # log_expYX = torch.log(1 + torch.exp(log_expYX))
            # part1 = Variable(W_selected * log_expYX, requires_grad=True)
            # part1 = part1.sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            part1 = W_selected * part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1
            # print('lambda3:', lambda3, 'lambda4:', lambda4)
            # print('L2:', L2, 'L1:', L1)
            return {
                'loss': loss,
                'loss_detail': torch.tensor([part1, L2, L1]),
                'prediction': prediction.argmax(1),
                'pred_orign': prediction
            }
        elif mode == 'eval':
            lambda3 = Vector_Classifier.causal_hyper['lambdas'][3]
            lambda4 = Vector_Classifier.causal_hyper['lambdas'][4]

            L1 = 0
            L2 = 0
            for param in self.parameters():
                L1 += param.abs().sum()
                L2 += (param ** 2).sum()

            part1 = (labels) * torch.log(prediction[:, 1]) + (1 - labels) * torch.log(prediction[:, 0])
            part1 = -part1
            part1 = part1.sum()

            # print('loss:', loss)
            loss = part1 + lambda3 * L2 + lambda4 * L1
            return {
                'prediction': prediction.argmax(1),
                'loss': loss
            }

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions

        return {
            'acc': (labels == preds).sum() / len(labels)
        }

    @staticmethod
    def set_causal_hyperparameter(w, lambdas):
        Vector_Classifier.causal_hyper = {
            'w': w,
            'lambdas': lambdas
        }
