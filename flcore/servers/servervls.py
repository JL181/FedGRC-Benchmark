import os
import time
import torch
from flcore.clients.clientvls import clientVLS
from flcore.servers.serverbase import Server
from utils.output_utils import round_ckpt_dir

class FedVLS(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientVLS)
        self.Budget = []
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')

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
            if getattr(self.args, 'ckpt_every', 0) and (i % self.args.ckpt_every == 0 or i == self.global_rounds):
                ckpt_dir = round_ckpt_dir(self.args)
                torch.save(self.global_model, os.path.join(ckpt_dir, f'round_{i:03d}.pt'))
            self.Budget.append(time.time() - s_t)
            completed_round_idx = i
            print(f'(Global Round {i}/{self.global_rounds}) completed time={self.Budget[-1]:.2f}s')
        self.send_models()
        print('Evaluate final personalized models')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self.save_results()
        self.save_global_model()
