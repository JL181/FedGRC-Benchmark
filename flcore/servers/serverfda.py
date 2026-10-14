import time
import torch
from flcore.clients.clientfda import clientFDA
from flcore.servers.serverbase import Server

class pFedFDA(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientFDA)
        self.Budget = []
        D = args.model.head.in_features
        C = args.num_classes
        self.global_means = torch.rand(C, D)
        self.global_covariance = torch.eye(D)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')

    def send_models(self):
        super().send_models()
        for client in self.selected_clients:
            client.global_means = self.global_means.clone()
            client.global_covariance = self.global_covariance.clone()

    def aggregate_parameters(self):
        super().aggregate_parameters()
        active = [self.clients[cid] for cid in self.uploaded_ids]
        total_samples = sum((c.train_samples for c in active))
        new_means = torch.zeros_like(self.global_means)
        new_cov = torch.zeros_like(self.global_covariance)
        for c in active:
            w = c.train_samples / total_samples
            new_means += w * c.adaptive_means.cpu()
            new_cov += w * c.adaptive_covariance.cpu()
        self.global_means = new_means
        self.global_covariance = new_cov

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f'\n-------------Round number: {i}-------------')
                print('Evaluate personalized models')
                self._fda_eval_all_clients()
                self.set_eval_context(i, 'pre_local_train')
                self.evaluate(round_idx=i)
            self.print_round_banner(i)
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_sel = len(self.selected_clients)
                train_info = client.train()
                print(f"(client {client_idx}/{total_sel}) client_id={client.id} done time={train_info['train_time']:.2f}s")
                self.print_client_loss_breakdown(client_idx, total_sel, client.id, train_info['loss_breakdown'])
            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            completed_round_idx = i
            print(f'(Global Round {i}/{self.global_rounds}) completed time={self.Budget[-1]:.2f}s')
        self.send_models()
        print('Setting LDA weights for final evaluation...')
        self._fda_eval_all_clients()
        print('Evaluate final personalized models')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self.save_results()
        self.save_global_model()

    def _fda_eval_all_clients(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.global_means = self.global_means.clone()
            client.global_covariance = self.global_covariance.clone()
            client.set_fda_eval()
