import math
import random
import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from flcore.clients.clientfa import clientFA
from flcore.servers.serverbase import Server
from utils.output_utils import flat_model_basename, round_ckpt_dir


def initialize_anchors(num_classes, feature_dim, seed, *, device, dtype=torch.float32):
    if num_classes < 1 or feature_dim < 1:
        raise ValueError('FedFA requires positive class and feature dimensions')
    if num_classes <= feature_dim:
        indices = random.Random(seed).sample(range(feature_dim), num_classes)
        anchors = torch.zeros(num_classes, feature_dim, dtype=dtype)
        anchors[torch.arange(num_classes), indices] = 1
    else:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        anchors = F.normalize(torch.randn(num_classes, feature_dim, generator=generator, dtype=dtype), dim=1)
    return anchors.to(device=device)


@torch.no_grad()
def aggregate_anchors(uploads, weights):
    if not uploads or len(uploads) != len(weights):
        raise ValueError('FedFA needs one anchor table per received model weight')
    weights = [float(w) for w in weights]
    if any((not math.isfinite(w) or w < 0 for w in weights)) or not math.isclose(sum(weights), 1, abs_tol=1e-06):
        raise ValueError('FedFA aggregation weights must be nonnegative and sum to one')
    result = torch.zeros_like(uploads[0])
    for (table, weight) in zip(uploads, weights):
        if table.shape != result.shape or table.device != result.device or table.dtype != result.dtype:
            raise ValueError('FedFA anchor uploads must share shape, dtype and device')
        result.add_(table.detach() * weight)
    if not torch.isfinite(result).all():
        raise FloatingPointError('FedFA aggregated anchors are not finite')
    return result


class FedFA(Server):

    def __init__(self, args, times):
        if getattr(args, 'backbone_ckpt', None) or getattr(args, 'train_backbone', False):
            raise ValueError('FedFA is a complete training method; closed-form-head flags do not apply')
        if args.global_rounds < 0 or args.eval_gap < 1 or (not 0 < args.join_ratio <= 1) or (int(args.num_clients * args.join_ratio) < 1):
            raise ValueError('FedFA requires gr>=0, eval_gap>=1 and at least one participating client')
        if not hasattr(args.model, 'base') or not isinstance(getattr(args.model, 'head', None), nn.Linear):
            raise ValueError('FedFA requires BaseHeadSplit with a linear head')
        for (key, default) in (('fedfa_mu', 0.1), ('fedfa_anchor_momentum', 0.5), ('fedfa_calibration_lr', 0.001)):
            setattr(args, key, float(getattr(args, key, default)))
        super().__init__(args, times)
        self.anchors = initialize_anchors(self.num_classes, self.global_model.head.in_features, args.seed, device=self.device, dtype=self.global_model.head.weight.dtype)
        self.set_slow_clients()
        self.set_clients(clientFA)
        self.clients_by_id = {c.id: c for c in self.clients}
        self.uploaded_anchors = []
        self.anchor_down_bytes = 0
        self.anchor_up_bytes = 0
        self.num_aggregations = 0
        self.Budget = []
        self.completed_round = None
        self.final_metrics = None
        print(f'[FedFA] anchors={tuple(self.anchors.shape)}')

    def send_models(self):
        super().send_models()
        for client in self.clients:
            client.set_anchors(self.anchors)
            count = self.payload_num_bytes(self.anchors)
            self.add_comm_bytes(count)
            self.anchor_down_bytes += count

    def receive_models(self):
        super().receive_models()
        if not self.uploaded_ids:
            raise RuntimeError('FedFA received no active client models; cannot aggregate')
        self.uploaded_anchors = []
        for cid in self.uploaded_ids:
            table = self.clients_by_id[cid].upload_anchors
            if table is None:
                raise RuntimeError(f'FedFA client {cid} uploaded a model without updated anchors')
            self.uploaded_anchors.append(table)
            count = self.payload_num_bytes(table)
            self.add_comm_bytes(count)
            self.anchor_up_bytes += count

    def aggregate_parameters(self):
        anchors = aggregate_anchors(self.uploaded_anchors, self.uploaded_weights)
        super().aggregate_parameters()
        self.anchors = anchors

    def training_state(self):
        return {'format_version': 1, 'anchors': self.anchors.detach().cpu().clone(), 'completed_round': self.completed_round, 'num_aggregations': self.num_aggregations, 'dataset': self.dataset, 'seed': self.args.seed, 'goal': self.goal, 'feature_dim': self.anchors.shape[1], 'num_classes': self.num_classes, 'hyperparameters': {key: getattr(self.args, key) for key in ('fedfa_mu', 'fedfa_anchor_momentum', 'fedfa_calibration_lr')}, 'anchor_down_bytes': self.anchor_down_bytes, 'anchor_up_bytes': self.anchor_up_bytes, 'comm_bytes_total': self.comm_bytes_total, 'resume_supported': False}

    def save_round_checkpoint(self, round_idx):
        directory = round_ckpt_dir(self.args)
        torch.save(self.global_model, os.path.join(directory, f'round_{round_idx:03d}.pt'))
        torch.save(self.training_state(), os.path.join(directory, f'round_{round_idx:03d}_fedfa_state.pt'))

    def save_global_model(self):
        super().save_global_model()
        stem = flat_model_basename(self.args)
        torch.save(self.training_state(), os.path.join(os.path.dirname(self.saved_model_path), f'{stem}_fedfa_state.pt'))

    def train(self):
        for i in range(self.global_rounds + 1):
            started = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i > 0 and i % self.eval_gap == 0:
                self.set_eval_context(i, 'pre_local_train')
                self.evaluate(round_idx=i)
            self.print_round_banner(i)
            for (number, client) in enumerate(self.selected_clients, 1):
                info = client.train()
                self.print_client_loss_breakdown(number, len(self.selected_clients), client.id, info['loss_breakdown'])
            self.receive_models()
            self.aggregate_parameters()
            elapsed = time.time() - started
            self.Budget.append(elapsed)
            self.completed_round = i
            self.num_aggregations += 1
            if self.args.ckpt_every and (i % self.args.ckpt_every == 0 or i == self.global_rounds):
                self.save_round_checkpoint(i)
            print(f'[FedFA] round {i}/{self.global_rounds} completed in {elapsed:.2f}s')
        self.send_models()
        self.set_eval_context(self.completed_round, 'post_train_final')
        self.final_metrics = self.evaluate(round_idx=self.completed_round)
        self.save_results()
        self.save_global_model()
