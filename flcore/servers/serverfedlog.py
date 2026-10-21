import os
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from flcore.clients.clientfedlog import clientFedLog, install_eta_into_head
from flcore.servers.serverbase import Server
from utils.data_utils import read_global_test_data
from utils.output_utils import round_ckpt_dir

def gauss_like_constant_func(eta, num_class, m):
    return torch.logsumexp((eta.view(num_class, m) ** 2).sum(dim=1) / 4, dim=0)


class ExponentialFamilyConjugatePrior:

    def __init__(self, norm_constant, chi, v=0):
        self.A = norm_constant
        self.chi_prior = chi.clone().detach()
        self.v_prior = v
        self.chi_posterior = chi.clone().detach()
        self.v_posterior = v
        self.m = chi.shape[0]
        self.kernel = lambda eta: -(eta @ self.chi_posterior / self.v_posterior - self.A(eta)) * self.v_posterior

    def update(self, chi, v):
        self.chi_posterior += chi
        self.v_posterior += v

    def forget(self):
        self.chi_posterior = self.chi_prior.clone().detach()
        self.v_posterior = self.v_prior

    def maximize(self, tol=0.001, max_iter=1000, lr=0.1, init=None):
        if init is None:
            eta = self.chi_posterior / self.v_posterior / 2
            eta.requires_grad = True
        else:
            eta = init
        opt = optim.Adam([eta], lr=lr)
        loss = None
        change = None
        for i in range(max_iter):
            opt.zero_grad()
            last = eta.detach().clone()
            loss = self.kernel(eta)
            loss.backward()
            opt.step()
            with torch.no_grad():
                change = torch.linalg.vector_norm(last - eta, ord=np.inf)
                if change <= tol:
                    print(i, loss.item(), change.item())
                    return (eta, str(change.item()))
        print(max_iter, loss.item(), change.item())
        return (eta, str(loss.item()))

class FedLog(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientFedLog)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating FedLog server and clients.')
        self.Budget = []
        if self.global_test_data is None:
            self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0 and (self.global_test_loader is None):
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)
        self.feat_dim = self.global_model.head.in_features
        self.m = self.feat_dim + 1
        self.m_total = self.m * self.num_classes
        self.map_iters = int(getattr(args, 'flog_map_iters', 10000))
        self.map_lr = float(getattr(args, 'flog_map_lr', 0.01))
        self.map_tol = float(getattr(args, 'flog_map_tol', 0.001))
        self.global_eta = None
        self.eta_prob = ExponentialFamilyConjugatePrior(norm_constant=lambda eta: gauss_like_constant_func(eta, self.num_classes, self.m), chi=torch.zeros(self.m_total, requires_grad=False).to(self.device))
        self.uploaded_ss = []
        self.uploaded_counts = []
        self.uploaded_n_cache = {}

    def charge_uplink(self, client_id, ss, n):
        extra = 0
        if self.uploaded_n_cache.get(client_id) != n:
            extra = 4
            self.uploaded_n_cache[client_id] = n
        self.add_comm_bytes(self.payload_num_bytes(ss) + extra)

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_global_head(self.global_eta)
            self.add_comm_bytes(self.payload_num_bytes(self.global_eta))
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_ss = []
        self.uploaded_counts = []
        for client in self.selected_clients:
            if client.upload_ss is None:
                continue
            self.uploaded_ids.append(client.id)
            self.uploaded_ss.append(client.upload_ss)
            self.uploaded_counts.append(client.upload_n)
            self.charge_uplink(client.id, client.upload_ss, client.upload_n)

    def aggregate_parameters(self):
        assert len(self.uploaded_ss) > 0
        self.eta_prob.forget()
        for (ss, n) in zip(self.uploaded_ss, self.uploaded_counts):
            self.eta_prob.update(chi=ss.to(self.device), v=n)
        self.solve_map()

    def solve_map(self):
        if self.eta_prob.v_posterior <= 0:
            if self.global_eta is None:
                raise RuntimeError('[FedLog] Cannot initialize the global head without training samples.')
            print('[FedLog] WARNING: no uploaded samples; skipping MAP and keeping the previous global head.')
            return None
        (eta, trace) = self.eta_prob.maximize(max_iter=self.map_iters, lr=self.map_lr, tol=self.map_tol)
        self.global_eta = eta.detach().view(self.num_classes, self.m).T.contiguous()
        if not torch.isfinite(self.global_eta).all():
            print('[FedLog] WARNING: non-finite entries in the solved global head.')
        install_eta_into_head(self.global_model.head, self.global_eta)
        return trace

    def pretrain(self):
        print('[FedLog] pretraining: statistics round on the initial bodies ...')
        self.eta_prob.forget()
        for client in self.clients:
            (ss, n) = client.compute_sufficient_statistics()
            self.eta_prob.update(chi=ss.to(self.device), v=n)
            self.charge_uplink(client.id, ss, n)
        self.solve_map()

    def train(self):
        self.pretrain()
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
        print('[FedLog] Saved the global head; the checkpoint backbone is untrained and client backbones are not included.')

    def evaluate(self, round_idx=None, acc=None, loss=None):
        return super().evaluate(round_idx=round_idx, acc=acc, loss=loss)

    def print_initial_setup(self):
        print(f"[FedLog] rounds={self.global_rounds} clients_per_round={self.current_num_join_clients} local_epochs={self.local_epochs} batch_size={self.batch_size} body_lr={getattr(self.args, 'flog_lr', 0.0005):g} map_lr={self.map_lr:g} map_iters={self.map_iters} map_tol={self.map_tol:g}")
