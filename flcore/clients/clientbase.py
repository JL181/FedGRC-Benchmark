import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

def resolve_sgd(args, default_momentum=0.0, default_weight_decay=0.0):
    m = getattr(args, 'momentum', None)
    w = getattr(args, 'weight_decay', None)
    return {'momentum': default_momentum if m is None else m, 'weight_decay': default_weight_decay if w is None else w}

class Client(object):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, **resolve_sgd(args))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for (new_param, old_param) in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        for (new_buf, old_buf) in zip(model.buffers(), self.model.buffers()):
            if new_buf.shape == old_buf.shape:
                old_buf.data = new_buf.data.clone()


    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for (x, y) in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return (test_acc, test_num, auc)

    def supports_generalization_detail(self):
        return True

    def detailed_personalized_test_metrics(self, testloader=None):
        testloaderfull = self.load_test_data() if testloader is None else testloader
        self.model.eval()
        correct = 0
        total = 0
        per_class_correct = [0 for _ in range(self.num_classes)]
        per_class_total = [0 for _ in range(self.num_classes)]
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        with torch.no_grad():
            for (x, y) in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                pred = torch.argmax(output, dim=1)
                matches = pred.eq(y)
                correct += matches.sum().item()
                total += y.shape[0]
                y_cpu = y.detach().cpu().tolist()
                pred_cpu = pred.detach().cpu().tolist()
                matches_cpu = matches.detach().cpu().tolist()
                for (label, predicted_label, is_correct) in zip(y_cpu, pred_cpu, matches_cpu):
                    per_class_total[label] += 1
                    per_class_correct[label] += int(is_correct)
                    confusion_matrix[label, predicted_label] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def detailed_generalization_test_metrics(self, testloader):
        return self.detailed_personalized_test_metrics(testloader=testloader)

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for (x, y) in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return (losses, train_num)


