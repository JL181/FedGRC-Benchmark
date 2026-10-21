import numpy as np
import time
from flcore.clients.clientbase import Client

class clientAVG(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'ce': 0.0}
        total_samples = 0
        for epoch in range(max_local_epochs):
            for (i, (x, y)) in enumerate(trainloader):
                (x, y) = self._prepare_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_size = y.shape[0]
                total_samples += batch_size
                loss_value = loss.item() * batch_size
                loss_sums['total'] += loss_value
                loss_sums['ce'] += loss_value
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed_time
        return {'local_epochs': max_local_epochs, 'train_time': elapsed_time, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def _prepare_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return (x, y)

    def _average_loss_sums(self, loss_sums, total_samples):
        if total_samples <= 0:
            return {key: 0.0 for key in loss_sums}
        return {key: value / total_samples for (key, value) in loss_sums.items()}
