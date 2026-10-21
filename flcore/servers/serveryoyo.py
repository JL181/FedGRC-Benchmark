import os
import time
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientyoyo import clientYoYo
from flcore.servers.serverbase import Server
from utils.data_utils import read_global_test_data
from utils.output_utils import round_ckpt_dir

def apply_change_threshold(cur, prev, threshold):
    eps = 1e-08
    change_ratio = torch.abs((cur - prev) / (prev + eps))
    revert_mask = change_ratio > threshold
    cur[revert_mask] = prev[revert_mask]
    return cur.clone()

class FedYoYo(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientYoYo)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        self.Budget = []
        self.global_eff_weights = torch.ones(args.num_classes)
        self.eff_ema_decay = 0.9
        self.completed_agg_rounds = 0
        if self.global_test_data is None:
            self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0 and (self.global_test_loader is None):
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)

    def send_models(self):
        super().send_models()
        for client in self.selected_clients:
            client.global_eff_weights = self.global_eff_weights.clone()

    def aggregate_eff_weights(self):
        active = [self.clients[cid] for cid in self.uploaded_ids]
        if len(active) == 0:
            return
        eff_cur = torch.ones_like(self.global_eff_weights)
        for c in active:
            eff_cur = eff_cur + c.local_eff_weights
        if self.completed_agg_rounds > 0:
            eff_cur = self.eff_ema_decay * self.global_eff_weights + (1 - self.eff_ema_decay) * eff_cur
            eff_cur = apply_change_threshold(eff_cur, self.global_eff_weights, threshold=100)
        else:
            abnormal = eff_cur > 10000.0
            if abnormal.any() and (~abnormal).any():
                mean_normal = eff_cur[~abnormal].mean()
                eff_cur[abnormal] = mean_normal
        self.global_eff_weights = eff_cur
        self.completed_agg_rounds += 1

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.current_round = i
            self.send_models()
            if i % self.eval_gap == 0:
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
            self.aggregate_eff_weights()
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
