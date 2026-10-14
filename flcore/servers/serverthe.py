import time
import torch
from flcore.clients.clientthe import clientTHE
from flcore.servers.serverbase import Server

class FedTHE(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientTHE)
        self.Budget = []
        rep_len = args.model.head.in_features
        self.global_rep = torch.zeros(rep_len)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')

    def send_models(self):
        super().send_models()
        for client in self.clients:
            client.global_rep = self.global_rep.clone().to(client.device)

    def aggregate_parameters(self):
        super().aggregate_parameters()
        active = [self.clients[cid] for cid in self.uploaded_ids]
        if len(active) > 0:
            new_rep = torch.zeros_like(self.global_rep)
            for c in active:
                new_rep += c.local_rep.cpu()
            self.global_rep = new_rep / len(active)

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f'\n-------------Round number: {i}-------------')
                print('Evaluate personalized models')
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
        print('Evaluate final personalized models')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self.save_results()
        self.save_global_model()
