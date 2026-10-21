import os
import time
import torch
from flcore.clients.clientdrplus import DotRegressionETFHead, build_dr_etf_weight, clientDrPlus
from flcore.servers.serverbase import Server
from utils.output_utils import round_ckpt_dir

def install_dr_etf_head(args):
    head = getattr(args.model, 'head', None)
    if head is None or not hasattr(head, 'weight'):
        raise NotImplementedError('FedDrPlus expects split_backbone_head(args) to have installed a linear head (args.model.head.weight); got head=%r.' % (type(head),))
    in_features = head.weight.shape[1]
    num_classes = head.weight.shape[0]
    etf_weight = build_dr_etf_weight(head.weight)
    etf_head = DotRegressionETFHead(in_features, num_classes).to(args.device)
    with torch.no_grad():
        etf_head.classifier.weight.copy_(etf_weight.to(args.device))
    args.model.head = etf_head
    print(f'[FedDrPlus] ETF classifier: d={in_features} C={num_classes} norm={etf_head.norm}')

class FedDrPlus(Server):

    def __init__(self, args, times):
        install_dr_etf_head(args)
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientDrPlus)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating FedDr+ server and clients.')
        self.Budget = []
        self.beta = float(getattr(args, 'drp_beta', 0.9))
        print(f'[FedDrPlus] beta={self.beta:g}')

    def print_client_loss_breakdown(self, client_idx, total_selected_clients, client_id, lb):
        print(f"(client {client_idx}/{total_selected_clients}) client_id={client_id} loss total={lb['total']:.4f} dr={lb['dr']:.4f} kd={lb['kd']:.4f}")

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print('Evaluate personalized models')
                self.set_eval_context(i, 'pre_local_train')
                self.evaluate(round_idx=i)
            self.print_round_banner(i)
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_selected_clients = len(self.selected_clients)
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} start train_samples={client.train_samples} test_samples={client.test_samples}')
                train_info = client.train()
                print(f"(client {client_idx}/{total_selected_clients}) client_id={client.id} done time={train_info['train_time']:.2f}s local_epochs={train_info['local_epochs']}")
                self.print_client_loss_breakdown(client_idx, total_selected_clients, client.id, train_info['loss_breakdown'])
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
