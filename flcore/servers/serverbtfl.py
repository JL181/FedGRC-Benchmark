import math
import os
import time
import torch
from flcore.clients.clientbtfl import clientBTFL
from flcore.servers.serverbase import Server
from utils.output_utils import round_ckpt_dir

class BTFL(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientBTFL)
        self.rep_len = args.model.head.in_features
        self.q_level = int(getattr(args, 'btfl_q_level', 2))
        self.global_dle = None
        self.global_reps = None
        self.Budget = []
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print(f"[BTFL] rep_len={self.rep_len} q_level={self.q_level} kb={getattr(args, 'btfl_kb', 1.0)} sigma={getattr(args, 'btfl_sigma', 0.01)} update_condition={getattr(args, 'btfl_update_condition', 5.0)} ent_norm={getattr(args, 'btfl_ent_norm', math.log(100)):.6f}")
        if 1 - 0.2 * self.q_level <= 0:
            print(f"[BTFL][WARNING] official CategoryDistributionModel smoothing is log(p * (1 - 0.2 * q_level) + 0.2) with q_level={self.q_level}, i.e. log(p * {1 - 0.2 * self.q_level:.4f} + 0.2). The argument is non-positive for p >= {(0.2 / (0.2 * self.q_level - 1) if 0.2 * self.q_level > 1 else float('inf')):.4f}, so the DLE log-likelihood becomes NaN and the per-sample gate degenerates. This is the official argparse default (parameters.py:297); every official exps/*.py overrides it with q_level=2. Use q_level=2 as documented in README.")
        print('Finished creating server and clients.')

    def exchange_dle(self):
        if len(self.selected_clients) == 0:
            return
        tables = []
        for client in self.selected_clients:
            table = client.compute_local_dle()
            self.add_comm_bytes(self.payload_num_bytes(table))
            tables.append(table)
        stacked = torch.stack(tables, dim=0)
        self.global_dle = stacked.mean(dim=0)
        for client in self.clients:
            client.receive_global_dle(self.global_dle)
            self.add_comm_bytes(self.payload_num_bytes(self.global_dle))

    def exchange_reps(self):
        if len(self.selected_clients) == 0:
            return
        reps = []
        for client in self.selected_clients:
            rep = client.local_rep.detach().cpu().clone()
            self.add_comm_bytes(self.payload_num_bytes(rep))
            reps.append(rep)
        self.global_reps = torch.stack(reps, dim=0)
        for client in self.clients:
            client.receive_global_reps(self.global_reps, len(reps))
            self.add_comm_bytes(self.payload_num_bytes(self.global_reps))

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0 and i > 0:
                print('Evaluate personalized models')
                self.set_eval_context(i, 'pre_local_train')
                self.evaluate(round_idx=i)
            self.print_round_banner(i)
            self.exchange_dle()
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_selected_clients = len(self.selected_clients)
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} start train_samples={client.train_samples} test_samples={client.test_samples}')
                train_info = client.train()
                print(f"(client {client_idx}/{total_selected_clients}) client_id={client.id} done time={train_info['train_time']:.2f}s local_epochs={train_info['local_epochs']}")
                self.print_client_loss_breakdown(client_idx, total_selected_clients, client.id, train_info['loss_breakdown'])
            self.exchange_reps()
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
