import random
import time
from torch.utils.data import DataLoader
from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
from utils.data_utils import read_global_test_data

class FedRep(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientRep)
        if self.global_test_data is None:
            self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0 and (self.global_test_loader is None):
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        self.Budget = []

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                if i > 0:
                    print('Evaluate personalized models')
                    self.set_eval_context(i, 'pre_local_train')
                    self.evaluate(round_idx=i)
            print(f"\n{'=' * 20} (Global Round {i}/{self.global_rounds}) {'=' * 20}")
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_sel = len(self.selected_clients)
                print(f'(client {client_idx}/{total_sel}) client_id={client.id} start train_samples={client.train_samples}')
                train_info = client.train()
                print(f"(client {client_idx}/{total_sel}) client_id={client.id} done time={train_info['train_time']:.2f}s local_epochs={train_info['local_epochs']}")
            self.receive_models()
            self.aggregate_parameters()
            round_time = time.time() - s_t
            self.Budget.append(round_time)
            completed_round_idx = i
            print(f'(Global Round {i}/{self.global_rounds}) completed time={round_time:.2f}s')
        self.send_models()
        print('Evaluate final personalized models')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            self.add_comm_bytes(self.payload_num_bytes(self.global_model))
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(1.0)
                self.uploaded_models.append(client.model.base)
                self.add_comm_bytes(self.payload_num_bytes(client.model.base))
        n = len(self.uploaded_weights)
        for i in range(n):
            self.uploaded_weights[i] = 1.0 / n
