import os
import time
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_global_test_data
from utils.output_utils import round_ckpt_dir

class FedAvg(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientAVG)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        self.Budget = []
        if self.global_test_data is None:
            self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0 and (self.global_test_loader is None):
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                if i == 0:
                    self.print_initial_setup()
                else:
                    print('Evaluate personalized models')
                    self.set_eval_context(i, 'pre_local_train')
                    self.evaluate(round_idx=i)
            self.print_round_banner(i)
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_selected_clients = len(self.selected_clients)
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} start train_samples={client.train_samples} test_samples={client.test_samples}')
                train_info = client.train()
                train_time = train_info['train_time']
                local_epochs = train_info['local_epochs']
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} done time={train_time:.2f}s local_epochs={local_epochs}')
                self.print_client_loss_breakdown(client_idx, total_selected_clients, client.id, train_info['loss_breakdown'])
            self.receive_models()
            self.aggregate_parameters()
            if getattr(self.args, 'ckpt_every', 0) and (i % self.args.ckpt_every == 0 or i == self.global_rounds):
                ckpt_dir = round_ckpt_dir(self.args)
                torch.save(self.global_model, os.path.join(ckpt_dir, f'round_{i:03d}.pt'))
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

    def evaluate(self, round_idx=None, acc=None, loss=None):
        return super().evaluate(round_idx=round_idx, acc=acc, loss=loss)

    def print_initial_setup(self):
        print(f'[{self.algorithm}] rounds={self.global_rounds} clients_per_round={self.current_num_join_clients} local_epochs={self.local_epochs} batch_size={self.batch_size} lr={self.learning_rate:g} eval_gap={self.eval_gap}')
