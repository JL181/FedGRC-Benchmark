from utils.checkpoint_utils import load_backbone
import os
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server
from utils.output_utils import round_ckpt_dir

def resolve_head_impl(model, error_message):
    for name in ('head', 'fc', 'classifier'):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module) and (not isinstance(module, nn.Identity)):
            return module
    raise AttributeError(error_message)

class ClosedFormHeadServer(Server):
    display_name = 'ClosedFormHead'

    def __init__(self, args, times, client_obj):
        self.backbone_ckpt = args.backbone_ckpt
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(client_obj)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print(f'Finished creating {self.display_name} server and clients.')
        self.Budget = []
        self.closed_form_fitted = False

    def resolve_head(self, model):
        return resolve_head_impl(model, f'{self.display_name} needs a model with a classifier head (BaseHeadSplit.head / .fc / .classifier); register {self.display_name} with split_backbone_head(args).')

    def feature_dim(self, model):
        head = self.resolve_head(model)
        in_features = getattr(head, 'in_features', None)
        if in_features is None:
            in_features = int(head.weight.shape[1])
        return int(in_features)

    def broadcast_fitted_head(self):
        head_bytes = self.payload_num_bytes(self.resolve_head(self.global_model))
        for client in self.clients:
            client.set_parameters(self.global_model)
            self.add_comm_bytes(head_bytes)

    @torch.no_grad()
    def install_affine_head(self, model, W, bias, require_bias=True):
        head = self.resolve_head(model)
        weight = torch.as_tensor(np.ascontiguousarray(W)).to(device=self.device, dtype=torch.float32)
        bias_t = torch.as_tensor(np.ascontiguousarray(bias)).to(device=self.device, dtype=torch.float32)
        if hasattr(head, 'install'):
            head.install(weight, bias_t)
            return
        if tuple(head.weight.shape) != tuple(weight.shape):
            raise RuntimeError(f'{self.display_name} head shape mismatch: head {tuple(head.weight.shape)} vs closed-form {tuple(weight.shape)}.')
        if getattr(head, 'bias', None) is None:
            if require_bias and float(torch.abs(bias_t).max()) > 0:
                raise RuntimeError(f'{self.display_name} produces a non-zero intercept but the head has no bias term.')
        head.weight.data.copy_(weight)
        if getattr(head, 'bias', None) is not None:
            head.bias.data.copy_(bias_t)

    @torch.no_grad()
    def collect_train_feature_bundle(self, src, fit_clients):
        from flcore.heads.extract_features import extract_client_train_features
        from flcore.heads.statistical_heads import derive_train
        src.eval()
        dev = torch.device(self.device)
        (ptf, pty, ptc) = ([], [], [])
        start = time.time()
        total_samples = 0
        for (pos, client) in enumerate(fit_clients):
            (feats, labels) = extract_client_train_features(src, self.dataset, client.id, dev)
            ptf.append(feats)
            pty.append(labels)
            ptc.append(np.full(len(labels), pos, dtype=np.int64))
            total_samples += len(labels)
            self.add_comm_bytes(feats.nbytes + labels.nbytes)
            if (pos + 1) % 20 == 0 or pos + 1 == len(fit_clients):
                print(f'[{self.display_name}] features {pos + 1}/{len(fit_clients)} clients ({total_samples} samples) {time.time() - start:.1f}s')
        return derive_train(np.concatenate(ptf), np.concatenate(pty), np.concatenate(ptc), int(self.num_classes), len(fit_clients))

    def train(self):
        if self.backbone_ckpt:
            self._train_attach()
            return
        self._train_selftrain_loop()

    def _train_selftrain_loop(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f'Evaluate models (SGD fc head monitor; {self.display_name} head is fitted after the loop)')
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
            round_time = time.time() - s_t
            self.Budget.append(round_time)
            completed_round_idx = i
            print(f'(Global Round {i}/{self.global_rounds}) completed time={round_time:.2f}s')
        print(f'\n------------- {self.display_name} closed-form classifier -------------')
        self.fit_closed_form_head()
        self.broadcast_fitted_head()
        print(f'Evaluate final models ({self.display_name} closed-form head)')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self._finalize_closed_form_run()

    def _train_attach(self):
        print(f'\n------------- {self.display_name} attach mode: closed-form head on a frozen backbone -------------')
        loaded = load_backbone(self.backbone_ckpt, self.global_model, self.args)
        resolve_head_impl(loaded, f'{self.display_name}: checkpoint has no classifier head')
        self.global_model = loaded.to(self.device)
        self.global_model.eval()
        print(f'[{self.display_name}] backbone loaded: {self.backbone_ckpt}')
        self.fit_closed_form_head()
        self.broadcast_fitted_head()
        print(f'Evaluate final models ({self.display_name} closed-form head)')
        self.set_eval_context(0, 'post_train_final')
        self.evaluate(round_idx=0)
        self._finalize_closed_form_run()


    def _finalize_closed_form_run(self):
        self.save_results()
        self.save_global_model()
